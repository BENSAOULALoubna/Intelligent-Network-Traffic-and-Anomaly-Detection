
import inet.node.inet.StandardHost;
import inet.node.ospfv2.OspfRouter;


module FtthLan
{
    parameters:
        int numHosts = default(5);
        @display("bgb=400,300");
    gates:
        inout uplinkGate;
    submodules:
        gateway: OspfRouter { @display("p=200,50"); }
        host[numHosts]: StandardHost { @display("p=50,180,row,60"); }
    connections:
        for i=0..numHosts-1 {
            host[i].ethg++ <--> LanAccessChannel <--> gateway.ethg++;
        }
        gateway.ethg++ <--> uplinkGate;
}