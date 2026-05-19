// DataExporter.h — rewritten with full feature set + TCP streaming
#ifndef DATAEXPORTER_H
#define DATAEXPORTER_H

#include <omnetpp.h>
#include <inet/common/packet/Packet.h>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <deque>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>

using namespace omnetpp;
using namespace inet;

// ─────────────────────────────────────────────────────────────────────────────
// Per-flow key for retransmission and RTT tracking
// ─────────────────────────────────────────────────────────────────────────────
struct FlowKey {
    std::string srcIp, dstIp;
    int dstPort;
    bool operator<(const FlowKey &o) const {
        if (srcIp   != o.srcIp)   return srcIp   < o.srcIp;
        if (dstIp   != o.dstIp)   return dstIp   < o.dstIp;
        return dstPort < o.dstPort;
    }
};

struct FlowState {
    simtime_t synTime      = SIMTIME_ZERO;
    bool      synSeen      = false;
    bool      synAckSeen   = false;
    bool      established  = false;
    int       retransmits  = 0;
    simtime_t lastSeq      = SIMTIME_ZERO;
};

// ─────────────────────────────────────────────────────────────────────────────
// Router-level stats — congestion dataset
// ─────────────────────────────────────────────────────────────────────────────
struct RouterStats {
    // throughput
    long bytes_in          = 0;
    long bytes_out         = 0;
    long packets_in        = 0;
    long packets_out       = 0;
    long packets_dropped   = 0;

    // protocol breakdown
    long ospf_packets      = 0;
    long arp_packets       = 0;
    long udp_packets       = 0;
    long tcp_packets       = 0;
    long icmp_packets      = 0;
    long tcp_ctrl_bytes    = 0;   // SYN/FIN/RST bytes
    long payload_bytes     = 0;
    long total_bytes       = 0;

    // TCP health
    long tcp_syn           = 0;
    long tcp_synack        = 0;
    long tcp_fin           = 0;
    long tcp_rst           = 0;
    long tcp_retransmits   = 0;
    long tcp_ack           = 0;

    // flow tracking
    std::set<std::string> active_flow_keys;  // "srcIp:dstIp:dstPort"
    int active_flows       = 0;
    long new_flows         = 0;
    long finished_flows    = 0;

    // inter-arrival jitter (Welford online variance on packet arrival times)
    simtime_t last_pkt_time = SIMTIME_ZERO;
    double    iat_mean     = 0.0;  // ms
    double    iat_M2       = 0.0;  // for Welford variance
    long      iat_count    = 0;

    // delay / RTT (from SYN→SYNACK round trips observed)
    double    rtt_sum      = 0.0;
    double    rtt_sq_sum   = 0.0;
    long      rtt_count    = 0;
    std::map<FlowKey, FlowState> flows;

    // burst detection: bytes in last 1s window (circular)
    std::deque<std::pair<simtime_t,long>> byte_window;  // (time, bytes)
    long      window_bytes = 0;

    // utilization reference set from NED param
    double    capacityMbps = 10000.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Observer-level stats — port scan dataset
// ─────────────────────────────────────────────────────────────────────────────
struct ObserverStats {
    // connection behavior
    long total_syn_count      = 0;
    long total_synack_count   = 0;
    long total_ack_count      = 0;
    long total_fin_count      = 0;
    long total_rst_count      = 0;
    long total_conn_attempts  = 0;
    long total_failed_conns   = 0;
    long total_completed_conn = 0;
    long tcp_ctrl_bytes       = 0;
    long payload_bytes        = 0;
    long total_packets        = 0;

    // diversity — unique endpoints (populated via header peek)
    std::set<std::string> unique_src_ips;
    std::set<std::string> unique_dst_ips;
    std::set<int>         unique_dst_ports;
    std::set<int>         unique_src_ports;

    // timing for IAT statistics
    std::vector<simtime_t> syn_times;     // for CV computation

    // lookback ring buffers (timestamp, running count)
    std::vector<std::pair<simtime_t,int>> syn_history;
    std::vector<std::pair<simtime_t,int>> ip_history;
    std::vector<std::pair<simtime_t,int>> port_history;

    // per-source SYN count (for fan-out: how many sources are scanning)
    std::map<std::string, int> syn_per_src;

    // scan rate: SYNs/s in current interval
    double syn_rate_per_sec   = 0.0;

    // jitter same as router
    simtime_t last_syn_time   = SIMTIME_ZERO;
    double    syn_iat_mean    = 0.0;
    double    syn_iat_M2      = 0.0;
    long      syn_iat_count   = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
class DataExporter : public cSimpleModule, public cListener
{
  private:
    cMessage      *collectionTimer = nullptr;
    std::ofstream  congestionStream;
    std::ofstream  portscanStream;

    // TCP socket for real-time dashboard
    int           socketFd = -1;
    std::string   dashboardHost = "127.0.0.1";
    int           dashboardPort = 5001;

    double         collectionInterval;
    std::string    congestionFile;
    std::string    portscanFile;
    std::string    scenarioName;
    int            congestionLabel;
    int            scanLabel;
    double         coreCapacityMbps;
    double         edgeCapacityMbps;

    std::map<std::string, RouterStats>   routerStats;
    std::map<std::string, ObserverStats> observerStats;

    simsignal_t packetSentSignal;
    simsignal_t packetReceivedSignal;
    simsignal_t packetDroppedSignal;

    // TCP socket helpers
    void setupTcpSocket();
    void sendJsonToDashboard(const std::string& json);
    void closeTcpSocket();

    // helpers
    void collectAndWrite();
    void writeCongestionRecord(const std::string &nodeId,
                               RouterStats &s, simtime_t t);
    void writeObserverRecord(const std::string &observerId,
                             ObserverStats &s, simtime_t t);
    void resetStats();
    std::string escapeJson(const std::string &s);
    int  getLookbackCount(const std::vector<std::pair<simtime_t,int>> &h,
                          double windowSeconds);
    void welfordUpdate(double newVal, double &mean, double &M2, long &n);
    double welfordVariance(double M2, long n);
    bool extractIpTcp(const Packet *pkt,
                      std::string &srcIp, std::string &dstIp,
                      int &srcPort, int &dstPort,
                      bool &isTcp, bool &isUdp,
                      bool &isSyn, bool &isSynAck,
                      bool &isAck, bool &isFin, bool &isRst,
                      long &payloadBytes);

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void receiveSignal(cComponent *src, simsignal_t id,
                               cObject *obj, cObject *details) override;
    virtual void finish() override;
};

#endif // DATAEXPORTER_H
