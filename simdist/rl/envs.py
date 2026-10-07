"""Look up environment configurations by the system name in ``config/system``."""


def get_record_env_cfg(system_name: str):
    if system_name == "go2":
        from simdist.rl.go2 import Go2RecordEnvCfg

        return Go2RecordEnvCfg()
    if system_name == "go1":
        from simdist.rl.go1 import Go1RecordEnvCfg

        return Go1RecordEnvCfg()
    raise ValueError(f"Unknown system: {system_name}")


def get_sim_env_cfg(system_name: str):
    if system_name == "go2":
        from simdist.rl.go2 import Go2SimEnvCfg

        return Go2SimEnvCfg()
    if system_name == "go1":
        from simdist.rl.go1 import Go1SimEnvCfg

        return Go1SimEnvCfg()
    raise ValueError(f"Unknown system: {system_name}")
