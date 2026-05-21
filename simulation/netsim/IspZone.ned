



module IspZone
{
    parameters:
        string zoneType;
        int numHosts;
    gates:
        inout uplinkGate;
    submodules:
        mobile: MobileLan if zoneType == "mobile" {
            parameters:
                numHosts = parent.numHosts;
        }
        ftth: FtthLan if zoneType == "ftth" {
            parameters:
                numHosts = parent.numHosts;
        }
        private: PrivateLan if zoneType == "private" {
            parameters:
                numHosts = parent.numHosts;
        }
    connections:
        if exists(mobile) {
            mobile.uplinkGate <--> uplinkGate;
        }
        if exists(ftth) {
            ftth.uplinkGate <--> uplinkGate;
        }
        if exists(private) {
            private.uplinkGate <--> uplinkGate;
        }
}
