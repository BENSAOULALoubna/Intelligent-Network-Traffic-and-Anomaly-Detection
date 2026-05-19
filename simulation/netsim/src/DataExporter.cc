// DataExporter.cc — full rewrite with rich feature extraction + TCP streaming
#include "DataExporter.h"
#include <inet/common/packet/Packet.h>
#include <inet/networklayer/ipv4/Ipv4Header_m.h>
#include <inet/transportlayer/tcp_common/TcpHeader_m.h>
#include <inet/transportlayer/contract/tcp/TcpCommand_m.h>
#include <inet/transportlayer/udp/UdpHeader_m.h>
#include <inet/networklayer/common/L3Address.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cerrno>
#include <cstring>

Define_Module(DataExporter);

// ─── TCP Socket Implementation ────────────────────────────────────────────────

void DataExporter::setupTcpSocket()
{
    // Close existing socket if any
    if (socketFd >= 0) {
        close(socketFd);
        socketFd = -1;
    }

    // Create socket
    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) {
        EV_WARN << "DataExporter: Failed to create TCP socket: " << strerror(errno) << "\n";
        return;
    }

    // Set non-blocking mode
    int flags = fcntl(socketFd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(socketFd, F_SETFL, flags | O_NONBLOCK);
    }

    // Configure dashboard address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(5001);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    // Try to connect (non-blocking)
    int result = connect(socketFd, (struct sockaddr*)&addr, sizeof(addr));
    if (result < 0 && errno != EINPROGRESS) {
        EV_WARN << "DataExporter: Failed to connect to dashboard: " << strerror(errno) << "\n";
        close(socketFd);
        socketFd = -1;
    } else {
        EV_INFO << "DataExporter: TCP socket created for dashboard at 127.0.0.1:5001\n";
    }
}

void DataExporter::sendJsonToDashboard(const std::string& json)
{
    if (socketFd < 0) {
        // Try to reconnect on next send
        setupTcpSocket();
        if (socketFd < 0) return;
    }

    std::string msg = json + "\n";
    ssize_t sent = send(socketFd, msg.c_str(), msg.length(), MSG_NOSIGNAL | MSG_DONTWAIT);

    if (sent < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            EV_WARN << "DataExporter: Dashboard connection lost, will retry\n";
            close(socketFd);
            socketFd = -1;
        }
    }
}

void DataExporter::closeTcpSocket()
{
    if (socketFd >= 0) {
        close(socketFd);
        socketFd = -1;
    }
}

// ─── Node name tables ────────────────────────────────────────────────────────
static const char* ROUTER_NAMES[] = {
    "boundary","core0","core1","core2","edge0","edge1", nullptr
};
static const char* OBSERVER_NAMES[] = {
    "isp_mobile0.mobile.gateway",
    "isp_mobile1.mobile.gateway",
    "isp_ftth0.ftth.gateway",
    "isp_private0.private.gateway",
    nullptr
};

// ─── initialize ──────────────────────────────────────────────────────────────
void DataExporter::initialize()
{
    collectionInterval = par("collectionInterval").doubleValue();
    congestionFile     = par("congestionFile").stdstringValue();
    portscanFile       = par("portscanFile").stdstringValue();
    scenarioName       = par("scenarioName").stdstringValue();
    congestionLabel    = par("congestionLabel").intValue();
    scanLabel          = par("scanLabel").intValue();
    coreCapacityMbps   = par("coreCapacityMbps").doubleValue();
    edgeCapacityMbps   = par("edgeCapacityMbps").doubleValue();

    // Read dashboard connection parameters from NED
    dashboardHost      = par("dashboardHost").stdstringValue();
    dashboardPort      = par("dashboardPort").intValue();

    congestionStream.open(congestionFile, std::ios::out | std::ios::app);
    portscanStream.open(portscanFile,     std::ios::out | std::ios::app);
    if (!congestionStream.is_open())
        throw cRuntimeError("Cannot open congestionFile: %s", congestionFile.c_str());
    if (!portscanStream.is_open())
        throw cRuntimeError("Cannot open portscanFile: %s", portscanFile.c_str());

    packetSentSignal     = registerSignal("packetSentToLower");
    packetReceivedSignal = registerSignal("packetReceivedFromLower");
    packetDroppedSignal  = registerSignal("packetDropped");

    getSimulation()->getSystemModule()->subscribe(packetSentSignal,     this);
    getSimulation()->getSystemModule()->subscribe(packetReceivedSignal, this);
    getSimulation()->getSystemModule()->subscribe(packetDroppedSignal,  this);

    for (int i = 0; ROUTER_NAMES[i];   i++) {
        routerStats[ROUTER_NAMES[i]] = RouterStats();
        // assign capacity reference: boundary/core = 10Gbps, others = 1Gbps
        std::string n = ROUTER_NAMES[i];
        routerStats[n].capacityMbps =
            (n=="boundary"||n.substr(0,4)=="core") ? coreCapacityMbps : edgeCapacityMbps;
    }
    for (int i = 0; OBSERVER_NAMES[i]; i++)
        observerStats[OBSERVER_NAMES[i]] = ObserverStats();

    collectionTimer = new cMessage("collectionTimer");
    scheduleAt(simTime() + collectionInterval, collectionTimer);

    // Setup TCP connection to dashboard
    setupTcpSocket();

    EV_INFO << "DataExporter initialized. interval=" << collectionInterval
            << "s, dashboard=" << dashboardHost << ":" << dashboardPort << "\n";
}

// ─── handleMessage ───────────────────────────────────────────────────────────
void DataExporter::handleMessage(cMessage *msg)
{
    if (msg == collectionTimer) {
        collectAndWrite();
        resetStats();
        scheduleAt(simTime() + collectionInterval, collectionTimer);
    }
}

// ─── header extraction ───────────────────────────────────────────────────────
bool DataExporter::extractIpTcp(const Packet *pkt,
    std::string &srcIp, std::string &dstIp,
    int &srcPort, int &dstPort,
    bool &isTcp, bool &isUdp,
    bool &isSyn, bool &isSynAck,
    bool &isAck, bool &isFin, bool &isRst,
    long &payloadBytes)
{
    isTcp=false; isUdp=false;
    isSyn=false; isSynAck=false; isAck=false; isFin=false; isRst=false;
    srcPort=0; dstPort=0; payloadBytes=0;
    srcIp=""; dstIp="";

    b offset = b(0);
    b totalLen = pkt->getTotalLength();

    while (offset < totalLen) {
        try {
            // Try IPv4 header
            auto ipChunk = pkt->peekAt<Ipv4Header>(offset);
            if (ipChunk) {
                srcIp = ipChunk->getSrcAddress().str();
                dstIp = ipChunk->getDestAddress().str();
                offset += ipChunk->getChunkLength();
                // Try TCP next
                try {
                    auto tcpChunk = pkt->peekAt<tcp::TcpHeader>(offset);
                    if (tcpChunk) {
                        isTcp    = true;
                        srcPort  = tcpChunk->getSrcPort();
                        dstPort  = tcpChunk->getDestPort();
                        isSyn    = tcpChunk->getSynBit() && !tcpChunk->getAckBit();
                        isSynAck = tcpChunk->getSynBit() &&  tcpChunk->getAckBit();
                        isAck    = tcpChunk->getAckBit() && !tcpChunk->getSynBit();
                        isFin    = tcpChunk->getFinBit();
                        isRst    = tcpChunk->getRstBit();
                        payloadBytes = std::max(0L, (long)(pkt->getByteLength()) - 40L);
                        return true;
                    }
                } catch (...) {}
                // Try UDP next
                try {
                    auto udpChunk = pkt->peekAt<UdpHeader>(offset);
                    if (udpChunk) {
                        isUdp    = true;
                        srcPort  = udpChunk->getSrcPort();
                        dstPort  = udpChunk->getDestPort();
                        payloadBytes = std::max(0L, (long)(pkt->getByteLength()) - 28L);
                        return true;
                    }
                } catch (...) {}
                return true; // got IP at least
            }
        } catch (...) {
            break;
        }
        break;
    }
    return false;
}


// ─── Welford online variance ─────────────────────────────────────────────────
void DataExporter::welfordUpdate(double newVal, double &mean, double &M2, long &n)
{
    n++;
    double delta  = newVal - mean;
    mean         += delta / n;
    double delta2 = newVal - mean;
    M2           += delta * delta2;
}
double DataExporter::welfordVariance(double M2, long n)
{
    return (n > 1) ? M2 / (n - 1) : 0.0;
}

// ─── receiveSignal ───────────────────────────────────────────────────────────
void DataExporter::receiveSignal(cComponent *src, simsignal_t id,
                                  cObject *obj, cObject *details)
{
    auto *pkt = dynamic_cast<Packet*>(obj);
    if (!pkt) return;

    std::string srcPath = src->getFullPath();
    long pktBytes = pkt->getByteLength();
    simtime_t now = simTime();

    // ── extract headers ──────────────────────────────────────────────────────
    std::string srcIp, dstIp;
    int srcPort=0, dstPort=0;
    bool isTcp=false, isUdp=false;
    bool isSyn=false, isSynAck=false, isAck=false, isFin=false, isRst=false;
    long payloadBytes=0;
    bool gotHeaders = false;
    try {
        gotHeaders = extractIpTcp(pkt, srcIp, dstIp, srcPort, dstPort,
                                   isTcp, isUdp, isSyn, isSynAck,
                                   isAck, isFin, isRst, payloadBytes);
    } catch (...) { /* header not available at this layer — skip */ }

    // ── name-based fallback for protocol classification ───────────────────────
    const char *pname = pkt->getName();
    bool isOspf = false, isArp = false, isIcmp = false;
    if (pname) {
        isOspf = (strstr(pname,"OSPF")||strstr(pname,"ospf"));
        isArp  = (strstr(pname,"arp") ||strstr(pname,"ARP"));
        isIcmp = (strstr(pname,"ICMP")||strstr(pname,"icmp")||strstr(pname,"ping"));
        if (!gotHeaders && !isOspf && !isArp && !isIcmp) {
            if (strstr(pname,"SYN")) { isTcp=true; isSyn=true; }
            else if (strstr(pname,"FIN")) { isTcp=true; isFin=true; }
            else if (strstr(pname,"RST")) { isTcp=true; isRst=true; }
            else if (strstr(pname,"ACK")) { isTcp=true; isAck=true; }
            else if (strstr(pname,"Udp")||strstr(pname,"udp")) isUdp=true;
            else if (strstr(pname,"Tcp")||strstr(pname,"tcp")) isTcp=true;
        }
    }
    bool isTcpCtrl = isTcp && (isSyn||isSynAck||isFin||isRst);

    // ── flow key ─────────────────────────────────────────────────────────────
    FlowKey fk;
    fk.srcIp   = srcIp.empty() ? "unknown" : srcIp;
    fk.dstIp   = dstIp.empty() ? "unknown" : dstIp;
    fk.dstPort = dstPort;
    std::string fkStr = fk.srcIp+":"+fk.dstIp+":"+std::to_string(fk.dstPort);

    // ═══════════════════════════════════════════════════════════════════════
    //  ROUTER STATS
    // ═══════════════════════════════════════════════════════════════════════
    for (int i = 0; ROUTER_NAMES[i]; i++) {
        if (srcPath.find(ROUTER_NAMES[i]) == std::string::npos) continue;
        RouterStats &rs = routerStats[ROUTER_NAMES[i]];

        if (id == packetSentSignal) {
            rs.bytes_out     += pktBytes;
            rs.packets_out   += 1;
            rs.total_bytes   += pktBytes;
            rs.payload_bytes += payloadBytes;
            rs.window_bytes  += pktBytes;
            // trim byte window (keep last 1s)
            rs.byte_window.push_back({now, pktBytes});
            simtime_t win_cutoff = now - 1.0;
            while (!rs.byte_window.empty() && rs.byte_window.front().first < win_cutoff) {
                rs.window_bytes -= rs.byte_window.front().second;
                rs.byte_window.pop_front();
            }
            if (isTcpCtrl) rs.tcp_ctrl_bytes += pktBytes;
            if (isOspf)    rs.ospf_packets++;
            if (isArp)     rs.arp_packets++;
            if (isIcmp)    rs.icmp_packets++;
            if (isTcp && !isTcpCtrl) rs.tcp_packets++;
            if (isUdp)     rs.udp_packets++;
            if (isSyn)     { rs.tcp_syn++; rs.new_flows++; }
            if (isSynAck)  rs.tcp_synack++;
            if (isFin)     { rs.tcp_fin++; rs.finished_flows++; }
            if (isRst)     { rs.tcp_rst++; rs.tcp_retransmits++; }
            if (isAck)     rs.tcp_ack++;

            // inter-arrival time jitter (Welford)
            if (rs.last_pkt_time != SIMTIME_ZERO) {
                double iat_ms = (now - rs.last_pkt_time).dbl() * 1000.0;
                welfordUpdate(iat_ms, rs.iat_mean, rs.iat_M2, rs.iat_count);
            }
            rs.last_pkt_time = now;

            // active flow tracking
            if (isSyn && !isSynAck) {
                rs.active_flow_keys.insert(fkStr);
                rs.active_flows = (int)rs.active_flow_keys.size();
            }
            if (isFin || isRst) {
                rs.active_flow_keys.erase(fkStr);
                rs.active_flows = (int)rs.active_flow_keys.size();
            }

            // RTT tracking via SYN→SYNACK
            if (isSyn && !isSynAck) {
                rs.flows[fk].synTime  = now;
                rs.flows[fk].synSeen  = true;
            }
            if (isSynAck && rs.flows.count(fk)) {
                FlowState &fs = rs.flows[fk];
                if (fs.synSeen && !fs.synAckSeen) {
                    double rtt_ms = (now - fs.synTime).dbl() * 1000.0;
                    rs.rtt_sum    += rtt_ms;
                    rs.rtt_sq_sum += rtt_ms * rtt_ms;
                    rs.rtt_count++;
                    fs.synAckSeen = true;
                }
            }
        }
        if (id == packetReceivedSignal) {
            rs.bytes_in    += pktBytes;
            rs.packets_in  += 1;
        }
        if (id == packetDroppedSignal) {
            rs.packets_dropped += 1;
        }


        break;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  OBSERVER STATS (ISP gateways — scan detection features)
    // ═══════════════════════════════════════════════════════════════════════
    for (int i = 0; OBSERVER_NAMES[i]; i++) {
        if (srcPath.find(OBSERVER_NAMES[i]) == std::string::npos) continue;
        ObserverStats &os = observerStats[OBSERVER_NAMES[i]];

        os.total_packets++;
        if (isTcpCtrl) os.tcp_ctrl_bytes += pktBytes;
        else           os.payload_bytes   += payloadBytes;

        // populate unique endpoint sets from extracted headers
        if (!srcIp.empty())  os.unique_src_ips.insert(srcIp);
        if (!dstIp.empty())  os.unique_dst_ips.insert(dstIp);
        if (dstPort > 0)     os.unique_dst_ports.insert(dstPort);
        if (srcPort > 0)     os.unique_src_ports.insert(srcPort);

        if (isSyn && !isSynAck) {
            os.total_syn_count++;
            os.total_conn_attempts++;
            os.syn_times.push_back(now);
            os.syn_history.push_back({now, (int)os.total_syn_count});
            os.ip_history.push_back({now,   (int)os.unique_dst_ips.size()});
            os.port_history.push_back({now, (int)os.unique_dst_ports.size()});
            if (!srcIp.empty()) os.syn_per_src[srcIp]++;

            // SYN inter-arrival time (Welford)
            if (os.last_syn_time != SIMTIME_ZERO) {
                double syn_iat_ms = (now - os.last_syn_time).dbl() * 1000.0;
                welfordUpdate(syn_iat_ms, os.syn_iat_mean, os.syn_iat_M2, os.syn_iat_count);
            }
            os.last_syn_time = now;
        }
        if (isSynAck)  os.total_synack_count++;
        if (isAck && !isSyn) os.total_ack_count++;
        if (isFin)     { os.total_fin_count++;  os.total_completed_conn++; }
        if (isRst)     { os.total_rst_count++;  os.total_failed_conns++; }
        break;
    }
}

// ─── collectAndWrite ─────────────────────────────────────────────────────────
void DataExporter::collectAndWrite()
{
    simtime_t t = simTime();
    for (int i = 0; ROUTER_NAMES[i];   i++)
        writeCongestionRecord(ROUTER_NAMES[i], routerStats[ROUTER_NAMES[i]], t);
    for (int i = 0; OBSERVER_NAMES[i]; i++)
        writeObserverRecord(OBSERVER_NAMES[i], observerStats[OBSERVER_NAMES[i]], t);
}

// ─── writeCongestionRecord ───────────────────────────────────────────────────
void DataExporter::writeCongestionRecord(const std::string &nodeId,
                                          RouterStats &s, simtime_t t)
{
    double dt = collectionInterval;

    // --- throughput ---
    double bw_in_mbps   = (s.bytes_in  * 8.0) / (dt * 1e6);
    double bw_out_mbps  = (s.bytes_out * 8.0) / (dt * 1e6);
    double util_in_pct  = (bw_in_mbps  / s.capacityMbps) * 100.0;
    double util_out_pct = (bw_out_mbps / s.capacityMbps) * 100.0;
    double bw_ratio     = (bw_in_mbps + bw_out_mbps) / 2.0;

    // --- loss ---
    long   total_pkt = s.packets_in + s.packets_out;
    double loss_pct  = total_pkt > 0 ?
        (double)s.packets_dropped / total_pkt * 100.0 : 0.0;

    // --- overhead ---
    double proto_oh_pct = s.total_bytes > 0 ?
        (double)(s.total_bytes - s.payload_bytes) / s.total_bytes * 100.0 : 0.0;
    double ospf_oh_pct  = s.total_bytes > 0 ?
        (double)(s.ospf_packets * 90) / s.total_bytes * 100.0 : 0.0;
    double tcp_ctrl_pct = s.total_bytes > 0 ?
        (double)s.tcp_ctrl_bytes / s.total_bytes * 100.0 : 0.0;

    // --- jitter ---
    double jitter_ms    = std::sqrt(welfordVariance(s.iat_M2, s.iat_count));

    // --- RTT ---
    double rtt_avg_ms   = s.rtt_count > 0 ? s.rtt_sum / s.rtt_count : 0.0;
    double rtt_var_ms   = s.rtt_count > 1 ?
        (s.rtt_sq_sum/s.rtt_count - rtt_avg_ms*rtt_avg_ms) : 0.0;
    double rtt_std_ms   = std::sqrt(std::max(0.0, rtt_var_ms));

    // --- burst (peak bytes/s in 1s window) ---
    double peak_bps     = (double)s.window_bytes * 8.0 / 1000.0; // kbps

    // --- TCP health ---
    double syn_rate     = s.tcp_syn   / dt;
    double rst_rate     = s.tcp_rst   / dt;
    double fin_rate     = s.tcp_fin   / dt;
    double retrans_rate = s.tcp_retransmits > 0 && s.tcp_syn > 0 ?
        (double)s.tcp_retransmits / s.tcp_syn * 100.0 : 0.0;

    // --- flow metrics ---
    double flow_churn   = (s.new_flows + s.finished_flows) / dt;

    // --- protocol mix ---
    double udp_frac = total_pkt > 0 ? (double)s.udp_packets / total_pkt : 0.0;
    double tcp_frac = total_pkt > 0 ? (double)s.tcp_packets / total_pkt : 0.0;

    std::ostringstream j;
    j << std::fixed << std::setprecision(6);
    j << "{"
      << "\"t\":"                           << t.dbl()              << ","
      << "\"node_id\":\""                   << escapeJson(nodeId)   << "\","
      << "\"scenario\":\""                  << escapeJson(scenarioName) << "\","
      << "\"type\":\"congestion\","
      // --- throughput & utilization ---
      << "\"bw_in_mbps\":"                  << bw_in_mbps           << ","
      << "\"bw_out_mbps\":"                 << bw_out_mbps          << ","
      << "\"bw_avg_mbps\":"                 << bw_ratio             << ","
      << "\"util_in_pct\":"                 << util_in_pct          << ","
      << "\"util_out_pct\":"                << util_out_pct         << ","
      // --- packet counts ---
      << "\"pkt_in\":"                      << s.packets_in         << ","
      << "\"pkt_out\":"                     << s.packets_out        << ","
      << "\"pkt_dropped\":"                 << s.packets_dropped    << ","
      << "\"pkt_loss_pct\":"                << loss_pct             << ","
      // --- bytes ---
      << "\"bytes_in\":"                    << s.bytes_in           << ","
      << "\"bytes_out\":"                   << s.bytes_out          << ","
      << "\"payload_bytes\":"               << s.payload_bytes      << ","
      << "\"total_bytes\":"                 << s.total_bytes        << ","
      // --- jitter & delay ---
      << "\"jitter_ms\":"                   << jitter_ms            << ","
      << "\"iat_mean_ms\":"                 << s.iat_mean           << ","
      << "\"rtt_avg_ms\":"                  << rtt_avg_ms           << ","
      << "\"rtt_std_ms\":"                  << rtt_std_ms           << ","
      // --- protocol breakdown ---
      << "\"ospf_pkts\":"                   << s.ospf_packets       << ","
      << "\"arp_pkts\":"                    << s.arp_packets        << ","
      << "\"icmp_pkts\":"                   << s.icmp_packets       << ","
      << "\"udp_pkts\":"                    << s.udp_packets        << ","
      << "\"tcp_pkts\":"                    << s.tcp_packets        << ","
      << "\"udp_fraction\":"                << udp_frac             << ","
      << "\"tcp_fraction\":"                << tcp_frac             << ","
      // --- overhead ---
      << "\"proto_overhead_pct\":"          << proto_oh_pct         << ","
      << "\"ospf_overhead_pct\":"           << ospf_oh_pct          << ","
      << "\"tcp_ctrl_pct\":"                << tcp_ctrl_pct         << ","
      // --- TCP health ---
      << "\"tcp_syn\":"                     << s.tcp_syn            << ","
      << "\"tcp_synack\":"                  << s.tcp_synack         << ","
      << "\"tcp_fin\":"                     << s.tcp_fin            << ","
      << "\"tcp_rst\":"                     << s.tcp_rst            << ","
      << "\"tcp_retransmit_pct\":"          << retrans_rate         << ","
      << "\"syn_rate_pps\":"                << syn_rate             << ","
      << "\"rst_rate_pps\":"                << rst_rate             << ","
      << "\"fin_rate_pps\":"                << fin_rate             << ","
      // --- flow tracking ---
      << "\"active_flows\":"                << s.active_flows       << ","
      << "\"new_flows\":"                   << s.new_flows          << ","
      << "\"finished_flows\":"              << s.finished_flows     << ","
      << "\"flow_churn_per_sec\":"          << flow_churn           << ","
      // --- burst ---
      << "\"peak_kbps_1s_window\":"         << peak_bps             << ","
      // --- label ---
      << "\"congestion_label\":"            << congestionLabel
      << "}";

    std::string jsonStr = j.str();
    congestionStream << jsonStr << "\n";
    congestionStream.flush();

    // Send to dashboard via TCP
    sendJsonToDashboard(jsonStr);
}

// ─── writeObserverRecord ─────────────────────────────────────────────────────
void DataExporter::writeObserverRecord(const std::string &observerId,
                                        ObserverStats &s, simtime_t t)
{
    double dt = collectionInterval;

    // --- ratios ---
    double syn_ack_ratio  = s.total_ack_count > 0 ?
        (double)s.total_syn_count / s.total_ack_count : (double)s.total_syn_count;
    double syn_rst_ratio  = s.total_rst_count > 0 ?
        (double)s.total_syn_count / s.total_rst_count : (double)s.total_syn_count;
    double success_pct    = s.total_conn_attempts > 0 ?
        (1.0 - (double)s.total_failed_conns / s.total_conn_attempts) * 100.0 : 100.0;
    double completion_pct = s.total_syn_count > 0 ?
        (double)s.total_completed_conn / s.total_syn_count * 100.0 : 100.0;

    // --- scan rate ---
    double syn_rate       = s.total_syn_count / dt;   // SYNs per second

    // --- fan-out ratio: unique_dst_ips per SYN (key scan indicator) ---
    int n_dst_ips   = (int)s.unique_dst_ips.size();
    int n_dst_ports = (int)s.unique_dst_ports.size();
    int n_src_ips   = (int)s.unique_src_ips.size();
    double fanout_ip   = s.total_syn_count > 0 ?
        (double)n_dst_ips   / s.total_syn_count : 0.0;
    double fanout_port = s.total_syn_count > 0 ?
        (double)n_dst_ports / s.total_syn_count : 0.0;

    // --- SYN IAT statistics ---
    double syn_iat_mean_ms = s.syn_iat_mean;
    double syn_iat_std_ms  = std::sqrt(welfordVariance(s.syn_iat_M2, s.syn_iat_count));
    // CV = stddev/mean — high CV = bursty, low CV with low mean = stealth scan
    double syn_iat_cv      = syn_iat_mean_ms > 0 ?
        syn_iat_std_ms / syn_iat_mean_ms : 0.0;

    // --- diversity ratio (unique_ports / unique_ips) ---
    double port_ip_ratio   = n_dst_ips > 0 ?
        (double)n_dst_ports / n_dst_ips : (double)n_dst_ports;

    // --- protocol overhead ---
    long total_b = s.tcp_ctrl_bytes + s.payload_bytes;
    double overhead_pct = total_b > 0 ?
        (double)s.tcp_ctrl_bytes / total_b * 100.0 : 0.0;

    // --- how many sources are scanning (distributed scan detection) ---
    int scanning_src_count = 0;
    for (auto &kv : s.syn_per_src)
        if (kv.second > 3) scanning_src_count++;  // threshold: >3 SYNs from one src

    // --- lookback windows ---
    int lb_syn_60   = getLookbackCount(s.syn_history,  60.0);
    int lb_syn_300  = getLookbackCount(s.syn_history,  300.0);
    int lb_ip_60    = getLookbackCount(s.ip_history,    60.0);
    int lb_port_60  = getLookbackCount(s.port_history,  60.0);

    std::ostringstream j;
    j << std::fixed << std::setprecision(6);
    j << "{"
      << "\"t\":"                             << t.dbl()                 << ","
      << "\"observer_node\":\""               << escapeJson(observerId)  << "\","
      << "\"scenario\":\""                    << escapeJson(scenarioName)<< "\","
      << "\"type\":\"portscan\","
      // --- connection counts ---
      << "\"total_syn\":"                     << s.total_syn_count       << ","
      << "\"total_synack\":"                  << s.total_synack_count    << ","
      << "\"total_ack\":"                     << s.total_ack_count       << ","
      << "\"total_fin\":"                     << s.total_fin_count       << ","
      << "\"total_rst\":"                     << s.total_rst_count       << ","
      << "\"conn_attempts\":"                 << s.total_conn_attempts   << ","
      << "\"failed_conns\":"                  << s.total_failed_conns    << ","
      << "\"completed_conns\":"               << s.total_completed_conn  << ","
      // --- rates ---
      << "\"syn_rate_per_sec\":"              << syn_rate                << ","
      // --- ratios ---
      << "\"syn_ack_ratio\":"                 << syn_ack_ratio           << ","
      << "\"syn_rst_ratio\":"                 << syn_rst_ratio           << ","
      << "\"conn_success_pct\":"              << success_pct             << ","
      << "\"conn_completion_pct\":"           << completion_pct          << ","
      // --- diversity ---
      << "\"unique_src_ips\":"                << n_src_ips               << ","
      << "\"unique_dst_ips\":"                << n_dst_ips               << ","
      << "\"unique_dst_ports\":"              << n_dst_ports             << ","
      << "\"fanout_ip_ratio\":"               << fanout_ip               << ","
      << "\"fanout_port_ratio\":"             << fanout_port             << ","
      << "\"port_ip_ratio\":"                 << port_ip_ratio           << ","
      // --- timing ---
      << "\"syn_iat_mean_ms\":"               << syn_iat_mean_ms         << ","
      << "\"syn_iat_std_ms\":"                << syn_iat_std_ms          << ","
      << "\"syn_iat_cv\":"                    << syn_iat_cv              << ","
      // --- overhead ---
      << "\"tcp_ctrl_bytes\":"                << s.tcp_ctrl_bytes        << ","
      << "\"payload_bytes\":"                 << s.payload_bytes         << ","
      << "\"total_packets\":"                 << s.total_packets         << ","
      << "\"proto_overhead_pct\":"            << overhead_pct            << ","
      // --- distributed detection ---
      << "\"scanning_src_count\":"            << scanning_src_count      << ","
      // --- lookback windows ---
      << "\"lb_syn_60s\":"                    << lb_syn_60               << ","
      << "\"lb_syn_300s\":"                   << lb_syn_300              << ","
      << "\"lb_dst_ip_60s\":"                 << lb_ip_60               << ","
      << "\"lb_dst_port_60s\":"               << lb_port_60             << ","
      // --- label ---
      << "\"scan_label\":"                    << scanLabel
      << "}";

    std::string jsonStr = j.str();
    portscanStream << jsonStr << "\n";
    portscanStream.flush();

    // Send to dashboard via TCP
    sendJsonToDashboard(jsonStr);
}

// ─── getLookbackCount ────────────────────────────────────────────────────────
int DataExporter::getLookbackCount(
    const std::vector<std::pair<simtime_t,int>> &history,
    double windowSeconds)
{
    simtime_t cutoff = simTime() - windowSeconds;
    int count = 0;
    for (auto &e : history)
        if (e.first >= cutoff) count++;
    return count;
}

// --- resetStats --------------------------------------------------------------
void DataExporter::resetStats()
{
    simtime_t cutoff300 = simTime() - 300.0;

    for (auto &kv : routerStats) {
        RouterStats &r = kv.second;
        double cap = r.capacityMbps;
        // preserve: active_flows, active_flow_keys, flows (RTT state), capacityMbps
        // preserve: iat_mean/M2/count for running jitter
        // preserve: byte_window / window_bytes (1s burst)
        auto saved_flows    = r.flows;
        auto saved_keys     = r.active_flow_keys;
        int  saved_af       = r.active_flows;
        auto saved_win      = r.byte_window;
        long saved_wb       = r.window_bytes;
        double saved_iat_m  = r.iat_mean;
        double saved_iat_M2 = r.iat_M2;
        long   saved_iat_n  = r.iat_count;
        simtime_t saved_lpt = r.last_pkt_time;
        r = RouterStats();
        r.capacityMbps    = cap;
        r.flows           = saved_flows;
        r.active_flow_keys= saved_keys;
        r.active_flows    = saved_af;
        r.byte_window     = saved_win;
        r.window_bytes    = saved_wb;
        r.iat_mean        = saved_iat_m;
        r.iat_M2          = saved_iat_M2;
        r.iat_count       = saved_iat_n;
        r.last_pkt_time   = saved_lpt;
    }

    for (auto &kv : observerStats) {
        ObserverStats &o = kv.second;
        // reset interval counters but keep lookback history and Welford state
        double saved_iat_m  = o.syn_iat_mean;
        double saved_iat_M2 = o.syn_iat_M2;
        long   saved_iat_n  = o.syn_iat_count;
        simtime_t saved_lpt = o.last_syn_time;
        auto syn_h = o.syn_history;
        auto ip_h  = o.ip_history;
        auto por_h = o.port_history;
        o = ObserverStats();
        o.syn_iat_mean  = saved_iat_m;
        o.syn_iat_M2    = saved_iat_M2;
        o.syn_iat_count = saved_iat_n;
        o.last_syn_time = saved_lpt;
        o.syn_history   = syn_h;
        o.ip_history    = ip_h;
        o.port_history  = por_h;
        // trim old entries beyond 300s lookback window
        auto trim = [&](std::vector<std::pair<simtime_t,int>> &v) {
            v.erase(std::remove_if(v.begin(), v.end(),
                [&](const std::pair<simtime_t,int> &e){ return e.first < cutoff300; }),
                v.end());
        };
        trim(o.syn_history);
        trim(o.ip_history);
        trim(o.port_history);
    }
}

// --- escapeJson --------------------------------------------------------------
std::string DataExporter::escapeJson(const std::string &s)
{
    std::string r; r.reserve(s.size());
    for (char c : s) {
        if      (c=='"')  r+="\\\"";
        else if (c=='\\') r+="\\\\";
        else if (c=='\n') r+="\\n";
        else if (c=='\r') r+="\\r";
        else if (c=='\t') r+="\\t";
        else              r+=c;
    }
    return r;
}

// --- finish ------------------------------------------------------------------
void DataExporter::finish()
{
    if (congestionStream.is_open()) congestionStream.close();
    if (portscanStream.is_open())   portscanStream.close();
    closeTcpSocket();
    if (collectionTimer) { cancelAndDelete(collectionTimer); collectionTimer=nullptr; }
}
