CPP_CODE = r"""#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <random>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <torch/torch.h>

std::mutex timing_lock;
struct ClientTiming {
    double training_s = 0.0, send_s = 0.0, recv_s = 0.0, comm_phase_s = 0.0;
};
std::map<int, ClientTiming> client_timing;

void add_timing(int client_id, const std::string& key, double delta_s) {
    if (delta_s <= 0) return;
    std::lock_guard<std::mutex> lock(timing_lock);
    if (key == "training_s") client_timing[client_id].training_s += delta_s;
    else if (key == "send_s") client_timing[client_id].send_s += delta_s;
    else if (key == "recv_s") client_timing[client_id].recv_s += delta_s;
    else if (key == "comm_phase_s") client_timing[client_id].comm_phase_s += delta_s;
}

const int BATCH_SIZE = 32;
const int EPOCHS_PER_ROUND = 1;
const double THRESHOLD = 0.6;
const int FIXED_DATA_PER_CLIENT = 5000;
torch::Device DEVICE(torch::kCPU);
int TIMEOUT = 25;
const int TCP_RETRIES = 3;
const int SERVER_BACKLOG = 128;
const int R_PRIME = 100;
const int MINIMUM_ROUNDS = 40;
const int COUNT_THRESHOLD = 5;

std::mutex msg_lck;
std::map<int, int> model_messages;
std::map<int, int> terminate_messages;
std::mutex latest_models_lock;
std::map<int, std::map<std::string, torch::Tensor>> latest_models;

bool _send_exact(int conn, const char* data, int n) {
    int w = 0;
    while(w < n) {
        int r = send(conn, data + w, n - w, MSG_NOSIGNAL);
        if(r <= 0) return false;
        w += r;
    }
    return true;
}
bool _recv_exact(int conn, char* buf, int n) {
    int r = 0;
    while(r < n) {
        int c = recv(conn, buf + r, n - r, 0);
        if(c <= 0) return false;
        r += c;
    }
    return true;
}

bool send_int(int conn, int v) {
    uint32_t val = htonl(v);
    return _send_exact(conn, (const char*)&val, 4);
}
bool send_str(int conn, const std::string& s) {
    if(!send_int(conn, s.size())) return false;
    return _send_exact(conn, s.data(), s.size());
}
bool send_tensor(int conn, const torch::Tensor& t) {
    std::ostringstream oss(std::ios::binary);
    torch::save(t, oss);
    return send_str(conn, oss.str());
}

int recv_int(int conn, bool& ok) {
    uint32_t val = 0;
    if(!_recv_exact(conn, (char*)&val, 4)) { ok = false; return 0; }
    ok = true;
    return ntohl(val);
}
std::string recv_str(int conn, bool& ok) {
    int sz = recv_int(conn, ok);
    if(!ok || sz == 0) return "";
    std::vector<char> buf(sz);
    if(!_recv_exact(conn, buf.data(), sz)) { ok = false; return ""; }
    return std::string(buf.data(), sz);
}
torch::Tensor recv_tensor(int conn, bool& ok) {
    std::string s = recv_str(conn, ok);
    if(!ok || s.empty()) { ok = false; return {}; }
    std::istringstream iss(s, std::ios::binary);
    torch::Tensor t;
    try { torch::load(t, iss); } catch(...) { ok = false; return {}; }
    return t;
}

struct NetMsg {
    std::string type;
    int id=0; int round=0; int terminate=0; int requester_id=0;
    std::vector<std::string> req_params;
    std::map<std::string, torch::Tensor> payload;
};

bool send_netmsg(int conn, const NetMsg& msg) {
    if(!send_str(conn, msg.type)) return false;
    if(!send_int(conn, msg.id)) return false;
    if(!send_int(conn, msg.round)) return false;
    if(!send_int(conn, msg.terminate)) return false;
    if(!send_int(conn, msg.requester_id)) return false;
    if(!send_int(conn, msg.req_params.size())) return false;
    for(const auto& rp : msg.req_params) if(!send_str(conn, rp)) return false;
    if(!send_int(conn, msg.payload.size())) return false;
    for(const auto& pair : msg.payload) {
        if(!send_str(conn, pair.first)) return false;
        if(!send_tensor(conn, pair.second)) return false;
    }
    return true;
}

bool recv_netmsg(int conn, NetMsg& msg) {
    bool ok;
    msg.type = recv_str(conn, ok); if(!ok) return false;
    msg.id = recv_int(conn, ok); if(!ok) return false;
    msg.round = recv_int(conn, ok); if(!ok) return false;
    msg.terminate = recv_int(conn, ok); if(!ok) return false;
    msg.requester_id = recv_int(conn, ok); if(!ok) return false;
    int nreq = recv_int(conn, ok); if(!ok) return false;
    for(int i=0; i<nreq; i++) {
        msg.req_params.push_back(recv_str(conn, ok));
        if(!ok) return false;
    }
    int npt = recv_int(conn, ok); if(!ok) return false;
    for(int i=0; i<npt; i++) {
        std::string k = recv_str(conn, ok); if(!ok) return false;
        torch::Tensor t = recv_tensor(conn, ok); if(!ok) return false;
        msg.payload[k] = t;
    }
    return true;
}

int NUM_CLIENTS = 0, NUM_MACHINES = 0;
std::string CURRENT_MACHINE_IP;
std::vector<std::string> ips;
struct Fault { int id, round, y; };
std::vector<Fault> faults;
std::vector<std::vector<int>> adj;

bool parse_input_file() {
    std::ifstream file("inputf.txt");
    if (!file.is_open()) return false;
    std::string line;
    std::getline(file, line);
    std::stringstream ss1(line);
    ss1 >> NUM_CLIENTS >> NUM_MACHINES;
    std::getline(file, line); CURRENT_MACHINE_IP = line;
    std::getline(file, line);
    std::stringstream ss2(line); std::string ip;
    while(std::getline(ss2, ip, ',')) ips.push_back(ip);
    std::getline(file, line);
    int num_faults = std::stoi(line);
    for(int i=0; i<num_faults; i++) {
        std::getline(file, line);
        std::stringstream ss3(line); std::string parts[3];
        std::getline(ss3, parts[0], ','); std::getline(ss3, parts[1], ','); std::getline(ss3, parts[2], ',');
        faults.push_back({std::stoi(parts[0]), std::stoi(parts[1]), std::stoi(parts[2])});
    }
    adj.resize(NUM_CLIENTS);
    for(int i=0; i<NUM_CLIENTS; i++)
        for(int j=0; j<NUM_CLIENTS; j++)
            if(i != j) adj[i].push_back(j);
    return true;
}

bool connect_with_timeout(int sock, const struct sockaddr *addr, socklen_t addrlen, int timeout_sec) {
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    int res = connect(sock, addr, addrlen);
    if(res < 0 && errno != EINPROGRESS) return false;
    if(res == 0) { fcntl(sock, F_SETFL, flags); return true; }
    fd_set fdset; FD_ZERO(&fdset); FD_SET(sock, &fdset);
    struct timeval tv; tv.tv_sec = timeout_sec; tv.tv_usec = 0;
    res = select(sock + 1, NULL, &fdset, NULL, &tv);
    if(res <= 0) return false;
    int so_error; socklen_t len = sizeof(so_error);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len);
    if(so_error != 0) return false;
    fcntl(sock, F_SETFL, flags);
    struct timeval rtv; rtv.tv_sec = 2; rtv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&rtv, sizeof(rtv));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&rtv, sizeof(rtv));
    return true;
}

class CIFAR10Dataset : public torch::data::datasets::Dataset<CIFAR10Dataset> {
    torch::Tensor images_;
    torch::Tensor targets_;
public:
    CIFAR10Dataset(const std::string& root) {
        std::vector<char> buffer;
        for (int i = 1; i <= 5; ++i) {
            std::string path = root + "/data_batch_" + std::to_string(i) + ".bin";
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) { std::cerr<<"File not found: "<<path<<"\n"; continue; }
            size_t size = file.tellg();
            if(size == 0) continue;
            file.seekg(0, std::ios::beg);
            std::vector<char> tmp(size);
            if (file.read(tmp.data(), size)) {
                buffer.insert(buffer.end(), tmp.begin(), tmp.end());
            }
        }
        int num_images = buffer.size() / 3073;
        if(num_images == 0) {
            std::cerr << "WARNING: Empty CIFAR10 dataset\n";
            num_images = 10;
            images_ = torch::randn({num_images, 3, 32, 32});
            targets_ = torch::zeros({num_images}, torch::kLong);
            return;
        }
        images_ = torch::empty({num_images, 3, 32, 32}, torch::kByte);
        targets_ = torch::empty({num_images}, torch::kLong);
        auto img_ptr = images_.data_ptr<uint8_t>();
        auto tgt_ptr = targets_.data_ptr<int64_t>();
        for (int i = 0; i < num_images; ++i) {
            tgt_ptr[i] = buffer[i * 3073];
            std::memcpy(img_ptr + i * 3072, buffer.data() + i * 3073 + 1, 3072);
        }
        images_ = images_.to(torch::kFloat32).div(255.0).sub(0.5).div(0.5);
    }
    torch::data::Example<> get(size_t index) override { return {images_[index], targets_[index]}; }
    torch::optional<size_t> size() const override { return images_.size(0); }
    torch::Tensor get_targets() { return targets_; }
};

class SubsetDataset : public torch::data::datasets::Dataset<SubsetDataset> {
    CIFAR10Dataset dataset_;
    std::vector<int> indices_;
public:
    SubsetDataset(CIFAR10Dataset dataset, std::vector<int> indices)
        : dataset_(std::move(dataset)), indices_(std::move(indices)) {}
    torch::data::Example<> get(size_t index) override { return dataset_.get(indices_[index]); }
    torch::optional<size_t> size() const override { return indices_.size(); }
};

std::vector<std::vector<int>> split_data(torch::Tensor targets, int num_clients, int fixed_data) {
    std::vector<std::vector<int>> client_indices(num_clients);
    std::map<int, std::vector<int>> class_indices;
    auto t_ptr = targets.data_ptr<int64_t>();
    for(int i=0; i<targets.size(0); i++) class_indices[t_ptr[i]].push_back(i);
    std::mt19937 gen(42);
    for(auto& pair: class_indices) {
        std::vector<int> idxs = pair.second;
        std::shuffle(idxs.begin(), idxs.end(), gen);
        int chunk = idxs.size() / num_clients;
        if(chunk == 0) continue;
        for(int c=0; c<num_clients; c++) {
            client_indices[c].insert(client_indices[c].end(), idxs.begin() + c*chunk, idxs.begin() + (c+1)*chunk);
        }
    }
    for(int c=0; c<num_clients; c++) {
        std::shuffle(client_indices[c].begin(), client_indices[c].end(), gen);
        if(client_indices[c].size() > fixed_data) client_indices[c].resize(fixed_data);
    }
    return client_indices;
}

// ============== MODELS ==============
struct SimpleCNNImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    SimpleCNNImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 6 * 6, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1(x)); x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2(x)); x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1(x)); return fc2(x);
    }
};

struct SimpleCNN10Impl : torch::nn::Module {
    torch::nn::Sequential features{nullptr}, classifier{nullptr};
    SimpleCNN10Impl() {
        features = register_module("features", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3,64,3).padding(1)), torch::nn::BatchNorm2d(64), torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).padding(1)), torch::nn::BatchNorm2d(64), torch::nn::ReLU(true),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64,128,3).padding(1)), torch::nn::BatchNorm2d(128), torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128,128,3).padding(1)), torch::nn::BatchNorm2d(128), torch::nn::ReLU(true),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128,256,3).padding(1)), torch::nn::BatchNorm2d(256), torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256,256,3).padding(1)), torch::nn::BatchNorm2d(256), torch::nn::ReLU(true),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256,512,3).padding(1)), torch::nn::BatchNorm2d(512), torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512,512,3).padding(1)), torch::nn::BatchNorm2d(512), torch::nn::ReLU(true),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
        ));
        classifier = register_module("classifier", torch::nn::Sequential(
            torch::nn::Linear(512*2*2, 256), torch::nn::ReLU(true), torch::nn::Dropout(0.5), torch::nn::Linear(256, 10)
        ));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x); x = x.view({x.size(0), -1}); return classifier->forward(x);
    }
};

torch::nn::Sequential make_vgg_layers(const std::vector<int>& cfg) {
    torch::nn::Sequential layers;
    int in_channels = 3;
    for(auto v : cfg) {
        if(v == -1) layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        else {
            layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, v, 3).padding(1)));
            layers->push_back(torch::nn::BatchNorm2d(v));
            layers->push_back(torch::nn::ReLU(true));
            in_channels = v;
        }
    }
    return layers;
}
struct VGGImpl : torch::nn::Module {
    torch::nn::Sequential features{nullptr}, classifier{nullptr};
    VGGImpl(torch::nn::Sequential f) {
        features = register_module("features", f);
        classifier = register_module("classifier", torch::nn::Sequential(
            torch::nn::Linear(512, 512), torch::nn::ReLU(true), torch::nn::Dropout(0.5),
            torch::nn::Linear(512, 512), torch::nn::ReLU(true), torch::nn::Dropout(0.5),
            torch::nn::Linear(512, 10)
        ));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x); x = x.view({x.size(0), -1}); return classifier->forward(x);
    }
};
std::shared_ptr<VGGImpl> VGG11BN() { return std::make_shared<VGGImpl>(make_vgg_layers({64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1})); }
std::shared_ptr<VGGImpl> VGG13BN() { return std::make_shared<VGGImpl>(make_vgg_layers({64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1})); }
std::shared_ptr<VGGImpl> VGG16BN() { return std::make_shared<VGGImpl>(make_vgg_layers({64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1})); }

struct _CifarBasicBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential shortcut{nullptr};
    _CifarBasicBlockImpl(int in_planes, int planes, int stride=1) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));
        if(stride != 1 || in_planes != planes) {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(planes)
            ));
        } else {
            shortcut = register_module("shortcut", torch::nn::Sequential());
        }
    }
    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));
        if(!shortcut->is_empty()) out += shortcut->forward(x);
        else out += x;
        return torch::relu(out);
    }
};

struct _CifarResNetImpl : torch::nn::Module {
    int in_planes = 16;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
    torch::nn::Linear fc{nullptr};
    
    torch::nn::Sequential _make_layer(int planes, int n, int stride) {
        torch::nn::Sequential layers;
        layers->push_back(std::make_shared<_CifarBasicBlockImpl>(in_planes, planes, stride));
        in_planes = planes;
        for(int i=1; i<n; i++) layers->push_back(std::make_shared<_CifarBasicBlockImpl>(planes, planes, 1));
        return layers;
    }
    _CifarResNetImpl(int n=3, int num_classes=10) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(16));
        layer1 = register_module("layer1", _make_layer(16, n, 1));
        layer2 = register_module("layer2", _make_layer(32, n, 2));
        layer3 = register_module("layer3", _make_layer(64, n, 2));
        fc = register_module("fc", torch::nn::Linear(64, num_classes));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = layer1->forward(out); out = layer2->forward(out); out = layer3->forward(out);
        out = torch::nn::functional::adaptive_avg_pool2d(out, torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1}));
        out = out.view({out.size(0), -1});
        return fc->forward(out);
    }
};


torch::Tensor forward_model(std::shared_ptr<torch::nn::Module> model, int model_choice, torch::Tensor x) {
    if(model_choice == 1) return std::dynamic_pointer_cast<SimpleCNNImpl>(model)->forward(x);
    if(model_choice == 2) return std::dynamic_pointer_cast<SimpleCNN10Impl>(model)->forward(x);
    if(model_choice >= 3 && model_choice <= 5) return std::dynamic_pointer_cast<VGGImpl>(model)->forward(x);
    if(model_choice == 6) return std::dynamic_pointer_cast<_CifarResNetImpl>(model)->forward(x);
    return x;
}

std::shared_ptr<torch::nn::Module> build_model(int choice) {
    if(choice == 1) return std::make_shared<SimpleCNNImpl>();
    if(choice == 2) return std::make_shared<SimpleCNN10Impl>();
    if(choice == 3) return VGG11BN();
    if(choice == 4) return VGG13BN();
    if(choice == 5) return VGG16BN();
    if(choice == 6) return std::make_shared<_CifarResNetImpl>(3, 10);
    return std::make_shared<SimpleCNNImpl>();
}
// ====================================

void broadcast_terminate(int id) {
    NetMsg msg; msg.type = "terminate";
    for(int pid : adj[id]) {
        terminate_messages[id]++;
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(8650 + pid);
        inet_pton(AF_INET, ips[pid].c_str(), &serv_addr.sin_addr);
        if(connect_with_timeout(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr), 1)) send_netmsg(sock, msg);
        close(sock);
    }
}

std::map<std::string, torch::Tensor> tcp_client_request_layers(int requester_id, int target_id, std::string target_ip, std::vector<std::string> param_names, int current_round, bool& ok_out) {
    int retries = TCP_RETRIES;
    ok_out = false;
    while(retries > 0) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1; setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in saddr; saddr.sin_family = AF_INET; saddr.sin_port = htons(8650 + target_id);
        inet_pton(AF_INET, target_ip.c_str(), &saddr.sin_addr);

        auto t0 = std::chrono::steady_clock::now();
        if(connect_with_timeout(sock, (struct sockaddr *)&saddr, sizeof(saddr), 1)) {
            { std::lock_guard<std::mutex> lck(msg_lck); model_messages[requester_id]++; }
            NetMsg msg; msg.type = "layer_request"; msg.requester_id = requester_id; msg.round = current_round; msg.req_params = param_names;
            send_netmsg(sock, msg);
            add_timing(requester_id, "send_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

            auto r0 = std::chrono::steady_clock::now();
            NetMsg resp; bool ok = recv_netmsg(sock, resp);
            add_timing(requester_id, "recv_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
            close(sock);
            if(ok && resp.type == "layer_response" && resp.round == current_round) {
                ok_out = true; return resp.payload;
            }
        } else close(sock);
        retries--;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return {};
}

void tcp_server(int id, std::atomic<bool>* stop_event, std::vector<int>* terminate_flags) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in address;
    address.sin_family = AF_INET; address.sin_addr.s_addr = INADDR_ANY; address.sin_port = htons(8650 + id);
    bind(server, (struct sockaddr *)&address, sizeof(address)); listen(server, SERVER_BACKLOG);

    struct timeval tv; tv.tv_sec = 1; tv.tv_usec = 0;
    setsockopt(server, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while(!*stop_event) {
        int conn = accept(server, nullptr, nullptr);
        if(conn < 0) continue;
        add_timing(id, "recv_s", 0.001);

        std::thread([id, conn, stop_event, terminate_flags]() {
            NetMsg msg;
            if(recv_netmsg(conn, msg)) {
                if(msg.type == "terminate") {
                    terminate_flags->push_back(1);
                    *stop_event = true;
                } else if(msg.type == "layer_request") {
                    std::map<std::string, torch::Tensor> snap;
                    { std::lock_guard<std::mutex> l(latest_models_lock);
                      if(latest_models.count(id)) snap = latest_models[id]; }

                    NetMsg resp; resp.type = "layer_response"; resp.id = id; resp.round = msg.round;
                    for(const auto& k : msg.req_params) if(snap.count(k)) resp.payload[k] = snap[k];
                    auto ts0 = std::chrono::steady_clock::now();
                    send_netmsg(conn, resp);
                    add_timing(id, "send_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count());
                }
            }
            close(conn);
        }).detach();
    }
    close(server);
}

std::string get_layer_key(std::string k) {
    size_t pos = k.find_last_of('.');
    if(pos != std::string::npos) return k.substr(0, pos);
    return k;
}

std::map<int, std::vector<int>> client_splits;
std::shared_ptr<CIFAR10Dataset> master_dataset;

void client_logic(int id, int model_choice) {
    auto dataset_train = SubsetDataset(*master_dataset, client_splits[id]).map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(dataset_train), torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(0));

    auto model = build_model(model_choice);
    model->to(DEVICE);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    int current_round = 0;
    std::atomic<bool> stop_event{false};
    std::vector<int> terminate_flags;
    std::vector<bool> crash_away_list(NUM_CLIENTS, false);
    
    std::thread server_thread(tcp_server, id, &stop_event, &terminate_flags);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    int counter = 0;

    while(current_round < R_PRIME) {
        auto ttrain = std::chrono::steady_clock::now();
        model->train();
        for(int ep=0; ep<EPOCHS_PER_ROUND; ep++) {
            for (auto& batch : *train_loader) {
                optimizer.zero_grad();
                auto out = forward_model(model, model_choice, batch.data.to(DEVICE));
                auto loss = torch::nn::CrossEntropyLoss()(out, batch.target.to(DEVICE));
                loss.backward();
                optimizer.step();
            }
        }
        add_timing(id, "training_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - ttrain).count());

        std::map<std::string, torch::Tensor> local_state_np;
        std::map<std::string, std::vector<std::string>> layer_groups;
        for(auto& p : model->named_parameters()) {
            local_state_np[p.key()] = p.value().detach().cpu().clone();
            layer_groups[get_layer_key(p.key())].push_back(p.key());
        }

        { std::lock_guard<std::mutex> l(latest_models_lock); latest_models[id] = local_state_np; }

        for(auto f : faults) {
            if(f.id == id && f.round == current_round) {
                std::cout << "Client " << id << " crashing at round " << current_round << std::endl;
                stop_event = true; server_thread.join(); return;
            }
        }
        if(!terminate_flags.empty()) { broadcast_terminate(id); break; }

        std::vector<int> alive;
        for(int n : adj[id]) if(!crash_away_list[n]) alive.push_back(n);
        alive.push_back(id);
        
        std::map<std::string, int> assignment;
        for(auto& pair : layer_groups) assignment[pair.first] = alive[rand() % alive.size()];

        std::map<int, std::vector<std::string>> params_needed;
        for(auto& pair : assignment) if(pair.second != id) {
            for(auto& n : layer_groups[pair.first]) params_needed[pair.second].push_back(n);
        }

        auto tcomm = std::chrono::steady_clock::now();
        std::map<int, std::map<std::string, torch::Tensor>> pulled;
        for(auto& pair : params_needed) {
            bool ok = false;
            auto resp = tcp_client_request_layers(id, pair.first, ips[pair.first], pair.second, current_round, ok);
            if(ok) pulled[pair.first] = resp;
            else crash_away_list[pair.first] = true;
        }
        add_timing(id, "comm_phase_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - tcomm).count());

        torch::NoGradGuard no_grad;
        for(auto& p : model->named_parameters()) {
            std::string n = p.key();
            std::string grp = get_layer_key(n);
            int chosen = assignment[grp];
            if(chosen != id && pulled.count(chosen) && pulled[chosen].count(n)) {
                p.value().copy_((p.value() + pulled[chosen][n].to(DEVICE)) / 2.0);
            }
        }

        std::cout << "Client " << id << " - Round " << current_round << " Done." << std::endl;
        
        if(current_round >= MINIMUM_ROUNDS) { 
            counter++;
            if(counter >= COUNT_THRESHOLD) { break; }
        }
        current_round++;
    }

    std::cout << "Client " << id << " finished." << std::endl;
    broadcast_terminate(id);
    stop_event = true;
    server_thread.join();
}

int main(int argc, char** argv) {
    if(!parse_input_file()) { std::cerr << "Failed to parse inputf.txt\n"; return 1; }
    
    int model_choice = 1;
    for(int i=1; i<argc-1; i++) {
        if(std::string(argv[i]) == "--model") model_choice = std::stoi(argv[i+1]);
    }
    std::cout << "Starting Federated Learning | Model: " << model_choice << "\n";

    master_dataset = std::make_shared<CIFAR10Dataset>("./data/cifar-10-batches-bin");
    auto splits = split_data(master_dataset->get_targets(), NUM_CLIENTS, FIXED_DATA_PER_CLIENT);
    for(int i=0; i<NUM_CLIENTS; i++) client_splits[i] = splits[i];
    
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for(int i=0; i<NUM_CLIENTS; i++) {
        if(ips[i] == CURRENT_MACHINE_IP) {
            workers.emplace_back(client_logic, i, model_choice);
        }
    }
    for(auto& t : workers) t.join();
    
    std::cout << "\nFederated Learning Completed\n";
    std::cout << "Current Machine IP : " << CURRENT_MACHINE_IP << "\n";
    std::cout << "Number of Clients  : " << NUM_CLIENTS << "\n";
    std::cout << "Total time taken   : " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " seconds\n";
    std::cout << "\nPer-client timing summary (seconds)\n";
    
    double agg_tr=0, agg_io=0, agg_ph=0, agg_tot=0;
    int local_c = 0;
    for(int i=0; i<NUM_CLIENTS; i++) {
        if(ips[i] == CURRENT_MACHINE_IP) {
            local_c++;
            auto t = client_timing[i];
            double io = t.send_s + t.recv_s;
            double tot = t.training_s + io + t.comm_phase_s;
            std::cout << "  Client "<<i<<": train="<<t.training_s<<", comm_io="<<io<<" [send "<<t.send_s<<", recv "<<t.recv_s<<"], comm_phase="<<t.comm_phase_s<<", comm_total="<<io+t.comm_phase_s<<", total="<<tot<<"\n";
            agg_tr+=t.training_s; agg_io+=io; agg_ph+=t.comm_phase_s; agg_tot+=tot;
        }
    }
    
    if(local_c > 0) {
        std::cout << "\nAggregate timing (local clients)\n";
        std::cout << "  Sum  : train="<<agg_tr<<", comm_io="<<agg_io<<", comm_phase="<<agg_ph<<", total="<<agg_tot<<"\n";
        std::cout << "  Avg  : train="<<agg_tr/local_c<<", comm_io="<<agg_io/local_c<<", comm_phase="<<agg_ph/local_c<<", total="<<agg_tot/local_c<<"\n";
    }

    return 0;
}
"""

with open("fedFaultRandLayers.cpp", "w") as f:
    f.write(CPP_CODE)
print("done")
