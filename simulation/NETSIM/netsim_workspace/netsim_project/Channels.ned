import inet.common.misc.ThruputMeteringChannel;

channel CoreBackboneChannel extends ThruputMeteringChannel
{
    datarate = 10Gbps;
    delay = 2ms;
    ber = 1e-9;
    thruputDisplayFormat = "u";
}

channel CoreToEdgeChannel extends ThruputMeteringChannel
{
    datarate = 1Gbps;
    delay = 5ms;
    ber = 1e-9;
    thruputDisplayFormat = "u";
}

channel EdgeToIspChannel extends ThruputMeteringChannel
{
    datarate = 1Gbps;
    delay = 8ms;
    ber = 1e-8;
    thruputDisplayFormat = "u";
}

channel IspToLanChannel extends ThruputMeteringChannel
{
    datarate = 1Gbps;
    delay = 10ms;
    ber = 1e-7;
    thruputDisplayFormat = "u";
}

channel LanAccessChannel extends ThruputMeteringChannel
{
    datarate = 100Mbps;
    delay = 0.1ms;
    ber = 1e-10;
    thruputDisplayFormat = "u";
}

channel BoundaryToCloudChannel extends ThruputMeteringChannel
{
    datarate = 10Gbps;
    delay = 1ms;
    ber = 1e-10;
    thruputDisplayFormat = "u";
}