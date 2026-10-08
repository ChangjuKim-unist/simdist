import flax.nnx as nnx
import jax.numpy as jnp

from simdist.modeling import types, scaler, encoders, modules
from simdist.utils import config, registry


_MODEL_REGISTRY: registry.Registry["ModelBase"] = registry.Registry("Model")


def register_model(name: str):
    return _MODEL_REGISTRY.register(name)


def get_model(
    cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs
) -> "ModelBase":
    model_name = cfg["model"]["type"]
    return _MODEL_REGISTRY.create(model_name, cfg, scaler_params, rngs)


class _ValueHead(nnx.Module):
    """Embedding, transformer encoder and decoder of a value head (frozen anchor copy)."""

    def __init__(self, emb, enc, dec):
        self.emb = emb
        self.enc = enc
        self.dec = dec


class ModelBase(nnx.Module):
    def __init__(
        self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs, **kwargs
    ):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.sys_cfg = cfg["system"]

    def __call__(
        self,
        x: types.ModelInputs,
        deterministic: bool | None = None,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement __call__ method")

    def inference(
        self,
        x: types.ModelInputs,
        **kwargs,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement inference method")


class WorldModelBase(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.proprio_obs_dim = config.proprio_obs_dim_from_sys_config(self.sys_cfg)
        self.extero_obs_dim = config.extero_obs_dim_from_sys_config(self.sys_cfg)
        self.action_dim = config.action_dim_from_sys_config(self.sys_cfg)
        self.cmd_dim = config.cmd_dim_from_sys_config(self.sys_cfg)
        self.latent_dim = self.model_cfg["latent_dim"]
        self.mlp_dropout_rate = self.model_cfg["dropout"]["mlp"]
        self.attention_dropout_rate = self.model_cfg["dropout"]["attention"]
        T = self.model_cfg["dataset"]["prediction_length"]
        emb_cfg = self.model_cfg["embedding"]
        emb_mlp_hsf = emb_cfg["mlp_hidden_size_factor"]
        emb_hidden_size = emb_mlp_hsf * self.latent_dim

        # input processing
        self.scaler = scaler.Scaler(
            scaler_params,
            types.WorldModelSchema.scaler_params_mapping,
        )
        self.encoder = encoders.WorldModelEncoderBase(cfg, rngs)
        self.fut_acts_embed = modules.Embedding(
            seq_len=T,
            input_dim=self.action_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["future_acts_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.fut_cmds_embed = modules.Embedding(
            seq_len=T,
            input_dim=self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["future_cmds_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )

        # dynamics
        attn_cfg = self.model_cfg["dynamics"]["attention"]
        self.dynamics = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )

        # reward head
        attn_cfg = self.model_cfg["reward"]["attention"]
        dec_cfg = self.model_cfg["reward"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.reward_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.action_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["reward_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.reward = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.reward_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        # value head
        self.value_emb, self.value, self.value_dec = self._build_value_head(rngs)

        # optional residual value head, V_real = V_sim + delta, zero-initialized so
        # adaptation starts exactly at the simulation value (see losses.WorldModelValueAdaptLoss)
        self.value_res = None
        res_cfg = self.model_cfg.get("value_residual", {})
        if res_cfg.get("enabled", False):
            self.value_res = self._build_value_residual(res_cfg, rngs)

        # optional frozen copy of the pretrained value head, used as the bootstrap
        # target during adaptation when the value head itself is being updated
        self.value_anchor = None
        if self.model_cfg.get("value_anchor", False):
            self.value_anchor = _ValueHead(*self._build_value_head(rngs))

        # policy head
        attn_cfg = self.model_cfg["policy"]["attention"]
        dec_cfg = self.model_cfg["policy"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.policy = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.policy_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=self.action_dim,
            rngs=rngs,
        )

    def _build_value_head(self, rngs: nnx.Rngs):
        T = self.model_cfg["dataset"]["prediction_length"]
        emb_cfg = self.model_cfg["embedding"]
        emb_hidden_size = emb_cfg["mlp_hidden_size_factor"] * self.latent_dim
        attn_cfg = self.model_cfg["value"]["attention"]
        dec_cfg = self.model_cfg["value"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        value_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["value_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        value = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        value_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        return value_emb, value, value_dec

    def _build_value_residual(self, res_cfg: dict, rngs: nnx.Rngs) -> modules.MLP:
        head = modules.MLP(
            input_dim=self.latent_dim + self.cmd_dim,
            hidden_dims=list(res_cfg.get("hidden_dims", [128, 128])),
            output_dim=1,
            rngs=rngs,
            dropout_rate=self.mlp_dropout_rate,
        )
        head.output_layer.kernel.value = jnp.zeros_like(head.output_layer.kernel.value)
        head.output_layer.bias.value = jnp.zeros_like(head.output_layer.bias.value)
        return head

    def add_value_residual(self, res_cfg: dict, rngs: nnx.Rngs) -> None:
        """Attach a zero-initialized residual value head to a pretrained model."""
        res_cfg = dict(res_cfg)
        res_cfg["enabled"] = True
        self.model_cfg["value_residual"] = res_cfg
        self.value_res = self._build_value_residual(res_cfg, rngs)

    def add_value_anchor(self) -> None:
        """Freeze a copy of the current value head as the adaptation bootstrap target."""
        self.model_cfg["value_anchor"] = True
        self.value_anchor = _ValueHead(
            nnx.clone(self.value_emb), nnx.clone(self.value), nnx.clone(self.value_dec)
        )

    def value_from_latents(
        self,
        latents: jnp.ndarray,
        fut_cmds_next: jnp.ndarray,
        deterministic: bool | None = None,
        use_residual: bool = True,
        use_anchor: bool = False,
    ) -> jnp.ndarray:
        """Value of each latent in ``latents`` (..., T, lat_dim), in scaled units.

        ``fut_cmds_next`` are the commands aligned with the latents (..., T, cmd_dim).
        With ``use_anchor`` the frozen pretrained head is used (no residual).
        """
        value_in = jnp.concatenate((latents, fut_cmds_next), axis=-1)
        if use_anchor:
            assert self.value_anchor is not None, "model has no value anchor"
            head = self.value_anchor
            value_in_emb = head.emb(value_in, deterministic=deterministic)
            value_pred = head.enc(value_in_emb, deterministic=deterministic)
            values = head.dec(value_pred, deterministic=deterministic)
        else:
            value_in_emb = self.value_emb(value_in, deterministic=deterministic)
            value_pred = self.value(value_in_emb, deterministic=deterministic)
            values = self.value_dec(value_pred, deterministic=deterministic)
            if use_residual and self.value_res is not None:
                values = values + self.value_res(value_in, deterministic=deterministic)
        return values.squeeze()

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Outputs:

        # pre-processing, encoding, and embedding
        x = self.scaler.scale(x)
        encoding = self.encoder(x, deterministic=deterministic)
        fut_acts_emb = self.fut_acts_embed(x["fut_acts"], deterministic=deterministic)
        fut_cmds_emb = self.fut_cmds_embed(
            x["fut_cmds"][:, :-1], deterministic=deterministic
        )

        # concatenate latent to the end of the history encoding
        latent_enc = jnp.expand_dims(encoding["latent"], axis=1)
        context = jnp.concatenate([encoding["history"], latent_enc], axis=1)

        # dynamics
        latents = self.dynamics(fut_acts_emb, context, deterministic=deterministic)

        # reward head
        # concatenate last latent with latent prediction
        z_t_tm1 = jnp.concatenate(
            (encoding["latent"][:, None, :], latents[:, :-1]), axis=1
        )
        # concatenate future actions and commands to latents
        rew_in = jnp.concatenate(
            (z_t_tm1, x["fut_acts"], x["fut_cmds"][:, :-1]), axis=-1
        )
        # embedding
        rew_in_emb = self.reward_emb(rew_in, deterministic=deterministic)
        # prediction
        rew_pred = self.reward(rew_in_emb, deterministic=deterministic)
        rewards = self.reward_dec(rew_pred, deterministic=deterministic).squeeze()

        # value head: latent prediction with future commands
        values = self.value_from_latents(latents, x["fut_cmds"][:, 1:], deterministic)

        # policy head
        latent_acts_pred = self.policy(
            fut_cmds_emb, context, deterministic=deterministic
        )
        actions = self.policy_dec(latent_acts_pred, deterministic=deterministic)

        return {
            "latents": latents,
            "rewards": rewards,
            "values": values,
            "actions": actions,
        }

    def inference(
        self,
        x: types.WorldModelSchema.Inputs,
    ) -> types.WorldModelSchema.Outputs:
        y = self(x, deterministic=True)
        y = self.scaler.unscale(y)
        return y

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """For latent dynamics consistency loss"""
        return self.encoder.encode_latent(
            proprio_obs, extero_obs, deterministic=deterministic
        )

    def get_scaler(self) -> scaler.Scaler:
        """
        Return the scaler.
        """
        return self.scaler


@register_model("quadruped_world_model")
class QuadrupedWorldModel(WorldModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.encoder = encoders.QuadrupedEncoder(cfg, rngs)
