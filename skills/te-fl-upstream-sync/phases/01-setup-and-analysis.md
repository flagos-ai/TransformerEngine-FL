### Stage 1: Repo Setup & Branch Preparation (`/stage1-setup`)

This stage gets the repo ready: clone, add upstream remote, and create the `dev` and `base` branches
needed for the merge. You can also run sub-steps individually: `/stage1-clone`, `/stage1-create-dev`,
`/stage1-create-base`.

#### Step 1: Clone the fork

```bash
git clone https://github.com/flagos-ai/TransformerEngine-FL.git
cd TransformerEngine-FL
```

Skip if already cloned — run the Repo Detection Preamble above instead.

#### Step 2: Add upstream remote and fetch

```bash
git remote -v | grep upstream
# If not present:
git remote add upstream https://github.com/Nvidia/TransformerEngine.git
git fetch upstream --tags
```

#### Step 3: Create dev branch from upstream release (`/stage1-create-dev`)

The `dev` branch mirrors the target upstream release exactly — no fork-specific changes.

**Before creating the dev branch, ask the user for two parameters:**

1. **Target upstream branch** — the upstream release to sync to (e.g. `release_v2.14`, `main`)
2. **Target upstream commit** — (optional) specific commit SHA on the target branch to checkout;
   if empty, use the branch tip

**Prompt the user:**
> Please specify:
> 1. Target upstream branch to sync to (e.g. `release_v2.14`)
> 2. Target upstream commit (leave empty for branch tip, e.g. `abc1234`)

Store the answers:
```bash
TARGET_UPSTREAM_BRANCH="<user answer 1>"   # e.g. release_v2.14
TARGET_UPSTREAM_COMMIT="<user answer 2>"   # e.g. abc1234 (empty = branch tip)
```

1. List available upstream releases for reference:
   ```bash
   git branch -r | grep upstream/release
   ```

2. Create the dev branch:
   ```bash
   if [ -n "$TARGET_UPSTREAM_COMMIT" ]; then
     git checkout -b dev ${TARGET_UPSTREAM_COMMIT}
     echo "Created dev at commit ${TARGET_UPSTREAM_COMMIT} on upstream/${TARGET_UPSTREAM_BRANCH}"
   else
     git checkout -b dev upstream/${TARGET_UPSTREAM_BRANCH}
     echo "Created dev at tip of upstream/${TARGET_UPSTREAM_BRANCH}"
   fi
   ```

3. Record the sync point — create `SYNC_POINT.md` at repo root:
   ```markdown
   # Upstream Sync Point
   - Upstream: Nvidia/TransformerEngine
   - Branch: ${TARGET_UPSTREAM_BRANCH}
   - Commit SHA: <output of `git rev-parse HEAD`>
   - Sync Date: <current date>
   - Synced By: <user>
   ```

4. Verify:
   ```bash
   git log --oneline -5
   ```

#### Step 4: Create base branch from fork's original upstream (`/stage1-create-base`)

The `base` branch represents the upstream version the fork was originally based on. This is needed
for accurate three-way merges.

**Ask the user for two more parameters:**

3. **Base upstream branch** — the upstream release the fork is currently based on (e.g. `release_v2.9`)
4. **Base upstream commit** — (optional) specific commit SHA on the base branch to checkout;
   if empty, use the branch tip

**Prompt the user:**
> Please specify:
> 3. Base upstream branch the fork is currently based on (e.g. `release_v2.9`)
> 4. Base upstream commit (leave empty for branch tip, e.g. `def5678`)

Store the answers:
```bash
BASE_UPSTREAM_BRANCH="<user answer 3>"     # e.g. release_v2.9
BASE_UPSTREAM_COMMIT="<user answer 4>"     # e.g. def5678 (empty = branch tip)
```

Create the base branch:
```bash
git fetch upstream ${BASE_UPSTREAM_BRANCH}

if [ -n "$BASE_UPSTREAM_COMMIT" ]; then
  git checkout -b base ${BASE_UPSTREAM_COMMIT}
  echo "Created base at commit ${BASE_UPSTREAM_COMMIT} on upstream/${BASE_UPSTREAM_BRANCH}"
else
  git checkout -b base upstream/${BASE_UPSTREAM_BRANCH}
  echo "Created base at tip of upstream/${BASE_UPSTREAM_BRANCH}"
fi

git log --oneline -5
```

**Success criteria:** `dev` and `base` branches exist and match their respective upstream releases
commit-for-commit.

---

### Stage 2: Identify Plugin Changes (`/stage2-diff-plugin-changes`)

Before merging, you need to understand exactly what the fork added on top of upstream. This diff
between `base` (the upstream release the fork is based on) and `main` (fork) reveals all plugin-related changes — the
files you must protect during the merge.

**Steps:**

1. Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

2. Generate a summary of all changes the fork introduced:
   ```bash
   git diff base..main --stat
   ```

3. Generate the full diff and save it for reference:
   ```bash
   git diff base..main > plugin_changes.diff
   ```

4. Identify plugin-specific changes:
   ```bash
   # Files added or modified in plugin directory
   git diff base..main --name-status -- 'transformer_engine/plugin/'

   # CUDA patches added or modified
   git diff base..main --name-status -- 'transformer_engine/__init__.py'

   # Build system changes for plugin support
   git diff base..main -- setup.py CMakeLists.txt pyproject.toml

   # API changes (e.g. torch.Tensor('cuda') -> torch.Tensor('TE_DEVICE_TYPE'))
   # Detailed changes captured by diff base..main
   git diff base..main --name-status -- 'transformer_engine/pytorch/'
   ```

5. Record the changes — save a structured summary to `PLUGIN_CHANGES.md`:
   ```markdown
   # Plugin Changes (base → main)

   ## New Files (added by fork)
   <list of files only in main, not in base>

   ## Modified Files (changed by fork)
   <list of files that exist in both but differ>

   ## Plugin Directory Contents
   <full listing of transformer_engine/common/plugin/>

   ## CUDA Patches Contents
   <full listing of transformer_engine/common/cuda_patches/>

   ## Build System Modifications
   <summary of plugin-related changes in setup.py, CMakeLists.txt, pyproject.toml>

   ## Python Binding Modifications
   <summary of plugin-related changes in transformer_engine/pytorch/>
   ```

This record is critical — during Stage 3 (Merge & Conflict Resolution), use it to verify that every
plugin change from main survives the merge. If a file listed here has a conflict, it needs
careful attention.

**Success criteria:** `plugin_changes.diff` and `PLUGIN_CHANGES.md` generated, all fork-specific
changes catalogued.

