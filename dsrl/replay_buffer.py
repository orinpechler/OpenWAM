import torch


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions for DSRL.

    Each transition is (obs, action, reward, next_obs, done) plus, optionally,
    the latent noise ``noise`` the latent actor fed to the frozen diffusion
    policy. One transition is one executed action chunk: ``action`` is the
    flattened chunk (the action critic's ``action_dim``), ``reward`` is the
    reward accumulated over that chunk, and ``next_obs`` is the observation
    after the chunk finished. ``done`` marks true termination only (task success
    or failure); time-limit truncation should be stored as ``done=False`` so the
    critic still bootstraps from ``next_obs``.

    Storage is preallocated on ``device``; once full, the oldest transitions are
    overwritten. ``sample`` draws uniformly with replacement.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        noise_dim: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.device = torch.device(device)
        self.dtype = dtype

        def empty(*shape: int) -> torch.Tensor:
            return torch.zeros(capacity, *shape, dtype=dtype, device=self.device)

        self.storage: dict[str, torch.Tensor] = {
            "obs": empty(obs_dim),
            "action": empty(action_dim),
            "reward": empty(),
            "next_obs": empty(obs_dim),
            "done": empty(),
        }
        if noise_dim is not None:
            self.storage["noise"] = empty(noise_dim)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _as_rows(self, value, width: int | None) -> torch.Tensor:
        """Convert ``value`` to a (N, width) tensor, or (N,) when ``width`` is None."""
        t = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if width is None:
            return t.reshape(-1)
        if t.numel() % width != 0:
            raise ValueError(f"cannot reshape tensor of shape {tuple(t.shape)} into rows of width {width}")
        # Flattens trailing dims, so an action chunk (N, horizon, dof) becomes (N, horizon * dof).
        return t.reshape(-1, width)

    def add(self, obs, action, reward, next_obs, done, noise=None) -> None:
        """Add one transition or a batch of N transitions.

        Inputs may be tensors, numpy arrays or scalars. A single transition has
        ``obs`` of shape (obs_dim,) and scalar ``reward``/``done``; a batch has a
        leading dimension N on every field. Trailing dimensions are flattened, so
        ``action`` may be passed as an unflattened chunk.
        """
        if (noise is None) != (self.noise_dim is None):
            raise ValueError("noise must be given if and only if the buffer was built with noise_dim")

        rows = {
            "obs": self._as_rows(obs, self.obs_dim),
            "action": self._as_rows(action, self.action_dim),
            "reward": self._as_rows(reward, None),
            "next_obs": self._as_rows(next_obs, self.obs_dim),
            "done": self._as_rows(done, None),
        }
        if noise is not None:
            rows["noise"] = self._as_rows(noise, self.noise_dim)

        n = rows["obs"].shape[0]
        if any(v.shape[0] != n for v in rows.values()):
            sizes = {k: v.shape[0] for k, v in rows.items()}
            raise ValueError(f"all fields must have the same number of transitions, got {sizes}")
        idx = (self._ptr + torch.arange(n, device=self.device)) % self.capacity
        if n > self.capacity:
            # Only the most recent `capacity` transitions would survive sequential adds.
            idx = idx[-self.capacity :]
            rows = {k: v[-self.capacity :] for k, v in rows.items()}
        for k, v in rows.items():
            self.storage[k][idx] = v
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Uniformly sample ``batch_size`` transitions (with replacement).

        Returns a dict with keys obs, action, reward, next_obs, done (and noise
        when stored), moved to ``device`` if given. ``generator`` must live on
        the buffer's device.
        """
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        idx = torch.randint(0, self._size, (batch_size,), device=self.device, generator=generator)
        batch = {k: v[idx] for k, v in self.storage.items()}
        if device is not None:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return batch

    def state_dict(self) -> dict:
        """Filled part of the buffer plus write position, for checkpointing."""
        return {
            "storage": {k: v[: self._size].cpu() for k, v in self.storage.items()},
            "ptr": self._ptr,
            "size": self._size,
        }

    def load_state_dict(self, state: dict) -> None:
        size = state["size"]
        if size > self.capacity:
            raise ValueError(f"checkpoint holds {size} transitions but capacity is {self.capacity}")
        if set(state["storage"]) != set(self.storage):
            raise ValueError(f"checkpoint fields {sorted(state['storage'])} != buffer fields {sorted(self.storage)}")
        for k, v in state["storage"].items():
            self.storage[k][:size] = v.to(device=self.device, dtype=self.dtype)
        self._ptr = state["ptr"] % self.capacity
        self._size = size
