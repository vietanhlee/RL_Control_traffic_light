from models.traffic_light.fixed_time import LightColor, SignalTiming


def test_cycle_time_sum():
    timing = SignalTiming(green=10, yellow=3, red=17)
    assert timing.cycle_time == 30


def test_light_state_rollover():
    timing = SignalTiming(green=4, yellow=2, red=4, offset=0)

    def color_at(t):
        cycle = timing.cycle_time
        phase = t % cycle
        if phase < timing.green:
            return LightColor.GREEN
        if phase < timing.green + timing.yellow:
            return LightColor.YELLOW
        return LightColor.RED

    assert color_at(1) == LightColor.GREEN
    assert color_at(4.5) == LightColor.YELLOW
    assert color_at(8) == LightColor.RED
