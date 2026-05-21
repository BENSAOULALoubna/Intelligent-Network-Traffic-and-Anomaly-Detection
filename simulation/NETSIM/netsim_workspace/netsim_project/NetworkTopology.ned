
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ospfv2.OspfRouter;
import inet.common.scenario.ScenarioManager;
import src.DataExporter;

network NetworkTopology
{
    parameters:
        int numHostsPerLan = default(5);
        @display("bgb=1200,900");

    submodules:
        configurator: Ipv4NetworkConfigurator { @display("p=50,50"); }
        scenarioManager: ScenarioManager      { @display("p=50,120"); }
        dataExporter: src.DataExporter            { @display("p=50,190"); }

        cloud: CloudNetwork   { @display("p=600,50"); }
        boundary: OspfRouter  { @display("p=600,180"); }
        core0: OspfRouter     { @display("p=380,320"); }
        core1: OspfRouter     { @display("p=600,320"); }
        core2: OspfRouter     { @display("p=820,320"); }
        edge0: OspfRouter     { @display("p=380,480"); }
        edge1: OspfRouter     { @display("p=820,480"); }

        isp_mobile0:  IspZone { numHosts=parent.numHostsPerLan; zoneType="mobile";  @display("p=180,650"); }
        isp_ftth0:    IspZone { numHosts=parent.numHostsPerLan; zoneType="ftth";    @display("p=430,650"); }
        isp_mobile1:  IspZone { numHosts=parent.numHostsPerLan; zoneType="mobile";  @display("p=680,650"); }
        isp_private0: IspZone { numHosts=parent.numHostsPerLan; zoneType="private"; @display("p=930,650"); }

    connections:
        boundary.ethg++ <--> BoundaryToCloudChannel <--> cloud.uplinkGate;
        boundary.ethg++ <--> CoreBackboneChannel <--> core0.ethg++;
        boundary.ethg++ <--> CoreBackboneChannel <--> core1.ethg++;
        boundary.ethg++ <--> CoreBackboneChannel <--> core2.ethg++;
        core0.ethg++ <--> CoreBackboneChannel <--> core1.ethg++;
        core1.ethg++ <--> CoreBackboneChannel <--> core2.ethg++;
        core2.ethg++ <--> CoreBackboneChannel <--> core0.ethg++;
        core0.ethg++ <--> CoreToEdgeChannel <--> edge0.ethg++;
        core1.ethg++ <--> CoreToEdgeChannel <--> edge0.ethg++;
        core1.ethg++ <--> CoreToEdgeChannel <--> edge1.ethg++;
        core2.ethg++ <--> CoreToEdgeChannel <--> edge1.ethg++;
        edge0.ethg++ <--> EdgeToIspChannel <--> isp_mobile0.uplinkGate;
        edge0.ethg++ <--> EdgeToIspChannel <--> isp_ftth0.uplinkGate;
        edge1.ethg++ <--> EdgeToIspChannel <--> isp_mobile1.uplinkGate;
        edge1.ethg++ <--> EdgeToIspChannel <--> isp_private0.uplinkGate;
}