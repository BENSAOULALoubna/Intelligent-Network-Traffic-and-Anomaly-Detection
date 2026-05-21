// DataExporter.ned
package src;

simple DataExporter
{
    parameters:
        double collectionInterval @unit(s) = default(10s);
        string congestionFile     = default("results/congestion_stream.jsonl");
        string portscanFile       = default("results/portscan_stream.jsonl");
        string scenarioName       = default("normal") @mutable;
        int    congestionLabel    = default(0) @mutable;
        int    scanLabel          = default(0) @mutable;
        double coreCapacityMbps   = default(10000.0);
        double edgeCapacityMbps   = default(1000.0);
        @display("i=block/sink");
        @class(DataExporter);
}