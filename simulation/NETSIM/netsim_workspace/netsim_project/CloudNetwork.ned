
import inet.node.inet.StandardHost;
import inet.node.ospfv2.OspfRouter;


module CloudNetwork
{
    parameters:
        @display("bgb=500,350");
    gates:
        inout uplinkGate;
    submodules:
        cloudGateway: OspfRouter { @display("p=250,50"); }
        videoServer: StandardHost { @display("p=60,180"); }
        voipServer: StandardHost { @display("p=160,180"); }
        webServer: StandardHost { @display("p=260,180"); }
        fileServer: StandardHost { @display("p=360,180"); }
        gameServer: StandardHost { @display("p=460,180"); }
    connections:
        videoServer.ethg++ <--> CoreBackboneChannel <--> cloudGateway.ethg++;
        voipServer.ethg++  <--> CoreBackboneChannel <--> cloudGateway.ethg++;
        webServer.ethg++   <--> CoreBackboneChannel <--> cloudGateway.ethg++;
        fileServer.ethg++  <--> CoreBackboneChannel <--> cloudGateway.ethg++;
        gameServer.ethg++  <--> CoreBackboneChannel <--> cloudGateway.ethg++;
        cloudGateway.ethg++ <--> uplinkGate;
}