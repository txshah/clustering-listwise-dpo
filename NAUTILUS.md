# Nautilus A6000 Setup

How to run this project on a Nautilus (NRP) RTX A6000 pod and attach to it from a laptop with herdr.
Each step is tagged with where it runs: **[laptop]**, **[pod]**, or **[browser]**.

The connection chain: `herdr --remote` → `ssh nautilus-a6000` → ProxyCommand (`kubectl exec` → `nc :22`) → sshd in the pod → herdr server working out of `/pvcvolume`.

The guiding rule: the container layer is disposable, the PVC is durable.
Everything that must survive an eviction (repo, venv, HF cache, ssh key, secrets) lives on `/pvcvolume`, and `pod-init.sh` rewires a fresh container around it.

## 1. Local prerequisites [laptop]

Check for an ssh key; generate one if the first command prints nothing:

```bash
ls ~/.ssh/*.pub
ssh-keygen -t ed25519
```

Install kubectl (macOS: `brew install kubectl`; Linux: grab the binary from `dl.k8s.io`) and herdr:

```bash
curl -fsSL https://herdr.dev/install.sh | sh
```

## 2. Nautilus kubeconfig [browser]

Log in at [portal.nrp-nautilus.io](https://portal.nrp-nautilus.io) and download the config (top-right menu → Get Config).
Save it as `~/.kube/config`, then:

```bash
chmod 600 ~/.kube/config
kubectl config set-context --current --namespace=<your-namespace>
kubectl get pods   # should answer without errors (empty is fine)
```

The whole ssh/herdr chain rides on kubectl auth.
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

Test it; this must work before herdr will:

```bash
ssh nautilus-a6000 echo ok   # prints: ok
```

Every new pod generates fresh host keys, so after an eviction ssh refuses with a MITM warning.
That is expected here; clear it with `ssh-keygen -R nautilus-a6000` and reconnect.

## 7. herdr in the pod, then attach [pod]

Install herdr inside the pod and park the binary on the PVC (already on PATH via `pod-init.sh`):

```bash
curl -fsSL https://herdr.dev/install.sh | sh
cp "$(command -v herdr)" /pvcvolume/bin/
```

Then from the laptop:

```bash
herdr --remote nautilus-a6000
```

Point the workspace at `/pvcvolume/clustering-listwise-dpo`.
Detach with `ctrl+b q`; agents and terminals keep running on the pod.
This replaces tmux for laptop-side disconnects.

## 8. After an eviction [laptop]

The short recovery path; everything of value was on the PVC:

```bash
kubectl get pods                # confirm the new pod is Running
kubectl exec -it deploy/assenthi-listwise-deployment -- bash /pvcvolume/clustering-listwise-dpo/pod-init.sh
ssh-keygen -R nautilus-a6000    # new pod = new host key
herdr --remote nautilus-a6000
```

Training or eval runs do not survive the eviction itself; relaunch them.
The pipeline scripts resume from what is on the PVC, and wandb shows the crashed run so you know where it died.
