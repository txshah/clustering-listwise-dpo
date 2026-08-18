# Nautilus A6000 Setup

How to run this project on a Nautilus (NRP) RTX A6000 pod, with ssh access and a persistent herdr session for long runs.
Each step is tagged with where it runs: **[laptop]**, **[pod]**, or **[browser]**.

The connection chain: `ssh nautilus-a6000` → ProxyCommand (`kubectl exec` → `nc :22`) → sshd in the pod → herdr session working out of `/pvcvolume`.

The guiding rule: the container layer is disposable, the PVC is durable.
Everything that must survive an eviction (repo, venv, HF cache, ssh key, secrets) lives on `/pvcvolume`, and `pod-init.sh` rewires a fresh container around it.

## 1. Local prerequisites [laptop]

Check for an ssh key; generate one if the first command prints nothing:

```bash
ls ~/.ssh/*.pub
ssh-keygen -t ed25519
```

Install kubectl (macOS: `brew install kubectl`; Linux: grab the binary from `dl.k8s.io`).

## 2. Nautilus kubeconfig [browser]

Log in at [portal.nrp-nautilus.io](https://portal.nrp-nautilus.io) and download the config (top-right menu → Get Config).
Save it as `~/.kube/config`, then:

```bash
chmod 600 ~/.kube/config
kubectl config set-context --current --namespace=<your-namespace>
kubectl get pods   # should answer without errors (empty is fine)
```

The whole ssh chain rides on kubectl auth.
If the remote ever goes unreachable weeks from now, re-download this config first.

## 3. Deploy PVC + A6000 pod [laptop]

```bash
git clone -b entailment-1000 git@github.com:txshah/clustering-listwise-dpo.git
cd clustering-listwise-dpo
kubectl apply -f pvc-listwise.yml
kubectl apply -f deployment-a6000.yml
kubectl get pods -w   # wait for Running; ctrl-c to stop watching
```

If it sits in Pending, A6000 nodes may be busy.
Check `kubectl describe pod <name>` for the scheduling reason, and verify the node label with `kubectl get nodes -L nvidia.com/gpu.product | grep -i a6000`.

## 4. Bootstrap the pod [laptop]

Push your public key onto the PVC, then run the repo's `pod-init.sh` in the pod (both commands from your laptop checkout):

```bash
cat ~/.ssh/id_ed25519.pub | kubectl exec -i deploy/assenthi-listwise-deployment -- sh -c 'cat > /pvcvolume/authorized_keys'
kubectl exec -i deploy/assenthi-listwise-deployment -- bash -s < pod-init.sh
```

If your key has a non-default name (e.g. `id_ed25519_personal.pub`), use that path here and add a matching `IdentityFile` line in step 6.

## 5. Project env on the PVC [pod]

One-time: shell in with `kubectl exec -it deploy/assenthi-listwise-deployment -- bash`, then build the env on the PVC.
The venv layers on the image's torch 2.4.1 (matching the repo pin), so torch is not reinstalled:

```bash
python -m venv --system-site-packages /pvcvolume/venv
source /pvcvolume/venv/bin/activate
pip install "transformers==4.45.1" "trl==0.9.6" "peft==0.13.2" \
    "accelerate==1.13.0" "datasets==4.8.4" pyyaml math-verify sentencepiece wandb

cd /pvcvolume && git clone -b entailment-1000 https://github.com/txshah/clustering-listwise-dpo.git

export HF_HOME=/pvcvolume/hf_cache
huggingface-cli login    # Mistral-7B is gated
```

For wandb, put the key in a secrets file on the PVC instead of `wandb login` (whose `~/.netrc` dies with the container):

```bash
echo 'export WANDB_API_KEY=<your key>' > /pvcvolume/secrets.env
chmod 600 /pvcvolume/secrets.env
```

`pod-init.sh` wires `~/.bashrc` to source the venv, `HF_HOME`, and `secrets.env` automatically.

## 6. ssh config entry [laptop]

Append to `~/.ssh/config`:

```
Host nautilus-a6000
    User root
    ProxyCommand kubectl exec -i deploy/assenthi-listwise-deployment -- nc localhost 22
    StrictHostKeyChecking accept-new
```

Test it:

```bash
ssh nautilus-a6000 echo ok   # prints: ok
```

Every new pod generates fresh host keys, so after an eviction ssh refuses with a MITM warning.
That is expected here; clear it with `ssh-keygen -R nautilus-a6000` and reconnect.

## 7. herdr in the pod [pod]

One-time: install herdr and park the binary on the PVC (`pod-init.sh` already put `/pvcvolume/bin` on PATH):

```bash
curl -fsSL https://herdr.dev/install.sh | sh
cp "$(command -v herdr)" /pvcvolume/bin/
```

Daily use, from the laptop:

```bash
ssh nautilus-a6000
herdr    # starts or reattaches to the pod's persistent session
```

Launch long runs inside it; point the workspace at `/pvcvolume/clustering-listwise-dpo`.
Detach with `ctrl+b q` (or just close the laptop; the herdr server keeps running).
Splits: `prefix+v` vertical, `prefix+minus` horizontal.

If you attach from a pane inside a local herdr, the outer one swallows `ctrl+b`.
Fix: set a different prefix (e.g. `[keys] prefix = "ctrl+a"`) in the pod's herdr config and save it as `/pvcvolume/herdr-config.toml`; `pod-init.sh` restores it to `~/.config/herdr/config.toml` on every boot.

## 8. After an eviction [laptop]

The short recovery path; everything of value was on the PVC:

```bash
kubectl get pods                # confirm the new pod is Running
kubectl exec -it deploy/assenthi-listwise-deployment -- bash /pvcvolume/clustering-listwise-dpo/pod-init.sh
ssh-keygen -R nautilus-a6000    # new pod = new host key
ssh nautilus-a6000
herdr
```

herdr sessions and running jobs do not survive the eviction itself; relaunch the runs.
The pipeline scripts resume from what is on the PVC, and wandb shows the crashed run so you know where it died.
