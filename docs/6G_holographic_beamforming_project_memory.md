# 6G Holographic Beamforming Dataset Project Memory

## Project objective

Build a 10,000-sample simulation dataset for training a deep-learning
model to learn/predict holographic beamforming behavior. The simulation
is based on the supplied phased-array + holographic beamforming code
using:

-   BS-side MMSE beamforming
-   Rayleigh fading RHS/user channel
-   Excitation matrix `E = genE(...)`
-   Gradient Projection (GP) beamforming
-   Sum-capacity / normalized mean rate as a system metric
-   Final beamforming matrices as target/output data

The dataset must preserve matrix dimensional consistency automatically.
No manual correction of individual samples is acceptable.

## Critical dimension decision

The original simulation logic must be respected.

### Fixed receiver dimension

``` python
Nr = 2
```

`Nr` must remain constant across all 10K samples.

### Transmit dimension

For `dmode = 1`, the original code defines:

``` python
Nx = int(lopPara[lop])
Nt = Nx ** 2
```

Therefore, `Nt` is structurally tied to the RHS array dimension.

For a machine-learning dataset where tensors must have identical
dimensions across all rows, use one fixed `Nx` and therefore one fixed:

``` python
Nt = Nx ** 2
```

Do NOT randomly sample `Nt` independently from `[32, 512]` while
simultaneously using `genE(lamd, deH, Nx, RFno)`. That creates
incompatible dimensions.

Recommended fixed dataset configuration:

``` python
Nr = 2
Nx = 16
Nt = Nx ** 2       # 256
```

If the exact original experiment uses another fixed `Nx`, replace only
`Nx`; keep:

``` python
Nr = 2
Nt = Nx ** 2
```

### RF dimension

`RFno` may vary only if `E` and `M` are constructed with the same RF
dimension for every sample. For a fixed-shape deep-learning target,
preferably keep:

``` python
RFno = 1
```

or select one fixed RF-chain count.

### L dimension

`L` appears in the channel tensor:

``` text
H : [test_size, Nr, Nt, L]
```

and in `w`:

``` text
w : [test_size, Nr, L]
```

Therefore, if a neural-network dataset requires fixed-size tensor
outputs, keep `L` fixed as well. A recommended value is:

``` python
L = 4
```

If `L` is varied, the raw beamforming/channel tensors cannot be directly
stacked without padding/masking.

## 6G frequency consideration

The project currently uses a research frequency range of 7--15 GHz.

The important distinction is that 7--15 GHz should be described as a
**6G research/simulation range**, not as a single finalized global 6G
standard band. ITU's IMT-2030 work is still defining/evaluating 6G radio
interfaces. Current Indian government information identifies/studies
portions of 7.125--8.4 GHz and 14.8--15.35 GHz for IMT/IMT-2030
consideration, with further decisions associated with WRC-27.

Sources:

-   ITU IMT-2030 technical requirements:
    https://www.itu.int/hub/2026/03/imt-2030-technical-requirements-for-the-6g-future/
-   ITU IMT-2030 framework/evaluation:
    https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx
-   Government of India 6G spectrum information:
    https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=2110684

For simulation:

``` python
freq = np.random.uniform(7e9, 15e9)
c = 3e8
lamd = c / freq
```

## Recommended 10K parameter generation

Use physically related parameters rather than independently random
values.

``` python
num_samples = 10000

freq_values = np.random.uniform(7e9, 15e9, num_samples)
lamd_values = 3e8 / freq_values

Nr_values = np.full(num_samples, 2, dtype=np.int32)

Nx_fixed = 16
Nt_fixed = Nx_fixed ** 2

Nx_values = np.full(num_samples, Nx_fixed, dtype=np.int32)
Nt_values = np.full(num_samples, Nt_fixed, dtype=np.int32)

L_values = np.full(num_samples, 4, dtype=np.int32)
RFno_values = np.full(num_samples, 1, dtype=np.int32)

SNR_values = np.random.uniform(0, 40, num_samples)

scale_values = np.random.uniform(0.5e9, 2e9, num_samples)

Pt_values = (
    10 ** (SNR_values / 10)
    * (1 / scale_values) ** 2
)

deH_values = lamd_values / 5.0
deP_values = lamd_values / 2.0
lamdAmp_values = deP_values / deH_values

test_size_values = np.random.randint(10, 50, num_samples)

stepSize1_values = np.random.uniform(1.0, 20.0, num_samples)

lopPara_values = np.random.choice(
    [10, 12, 14, 16],
    num_samples
)
```

## Noise parameter

Noise must be included as an explicit dataset input instead of being
hard-coded.

For example:

``` python
noise_var_values = 10 ** (
    -np.random.uniform(0, 20, num_samples) / 10
)
```

Then use:

``` python
W_MMSE = MMSE_Beamforming(
    H_BS,
    noise_var=noise_var
)
```

This makes noise variance part of the learned input space.

Do not use a fixed:

``` python
noise_var=0.1
```

if the objective is to train a model that generalizes across
channel/noise conditions.

## Mandatory tensor shapes

For every sample:

``` text
H_BS
    [test_size, Nt, Nt]

W_MMSE
    [test_size, Nt, Nt]

H_BS_W
    [test_size, Nt, Nt]

H_RHS
    [test_size, Nr, Nt, L]

H_RHS_fixed
    [test_size, L, Nr, Nt]

H_testD
    [test_size, L, Nr, Nt]

E0
    [1, Nt, RFno]

E
    [test_size, Nt, RFno]

m
    [test_size, Nt, RFno]

M
    [test_size, Nt, RFno]

w
    [test_size, Nr, L]
```

The GP implementation must never change `Nt`, `Nr`, `L`, or `RFno`
internally.

## Important GP correction

The previous GP versions repeatedly attempted to solve shape mismatches
by:

-   `repeat`
-   `expand`
-   trimming
-   arbitrary denominator repetition
-   arbitrary E permutation/repetition

This should NOT be used as the primary solution.

The correct principle is:

``` text
H and E must already be dimensionally compatible.
GP should operate on those fixed dimensions.
```

In particular, avoid code such as:

``` python
denominator = denominator.repeat(...)
```

when the repeat count is hard-coded.

Avoid:

``` python
m = m.expand(...)
```

for non-singleton dimensions.

Avoid arbitrary:

``` python
E_transformed = E_transformed.repeat(...)
```

to force incompatible antenna/RF dimensions.

These operations can make the code execute while changing the
mathematical meaning of the beamforming algorithm.

## Dataset outputs

Each simulation row should contain:

### Scalar inputs

``` text
sample_id
frequency_GHz
wavelength
Nr
Nx
Nt
L
RFno
SNR_dB
noise_var
scale
Pt
deH
deP
lamdAmp
test_size
stepSize1
lopPara
```

### Intermediate scalar features

Store numerical summaries of:

``` text
W_MMSE
H_BS_W
H_testD
E
initial_M
final_M
w
```

Recommended features:

``` text
W_MMSE_norm
W_MMSE_mean_abs
W_MMSE_std_abs

H_BS_W_norm
H_BS_W_mean_abs
H_BS_W_std_abs

H_testD_norm
H_testD_mean_abs
H_testD_std_abs

E_norm
E_mean_abs
E_std_abs

M_norm
M_mean_abs
M_std_abs

w_norm
w_mean_abs
w_std_abs
```

### Final outputs

At minimum:

``` text
Normalized_Mean_Rate
Execution_Time
```

For deep-learning beamforming prediction, also preserve:

``` text
w
M
```

as separate tensor files, preferably `.pt` or `.npz`, rather than
putting huge complex matrices directly into CSV.

## Recommended storage architecture

Do NOT put a complete complex beamforming matrix into every CSV cell if
the matrix is large.

Use:

``` text
6G_dataset/
    metadata.csv
    inputs.npz
    targets_w.npz
    targets_M.npz
```

or:

``` text
6G_dataset/
    metadata.csv
    tensors/
        sample_00000.pt
        sample_00001.pt
        ...
```

CSV should contain scalar features and filenames/IDs pointing to tensor
targets.

Example:

``` text
sample_id,frequency_GHz,Nr,Nx,Nt,L,RFno,SNR_dB,noise_var,...,Normalized_Mean_Rate,w_file,M_file
0,8.23,2,16,256,4,1,21.4,...,rate,... 
```

## Random seed

For reproducibility:

``` python
SEED = 42

np.random.seed(SEED)
t.manual_seed(SEED)
```

If GPU is used:

``` python
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
```

## Safety checks before GP

Every sample should execute:

``` python
assert H_testD.ndim == 4
assert H_testD.shape[1] == L
assert H_testD.shape[2] == Nr
assert H_testD.shape[3] == Nt

assert E.ndim == 3
assert E.shape[1] == Nt
assert E.shape[2] == RFno
```

Inside GP:

``` python
assert H.shape[2] == E.shape[1]
assert E.shape[1] == Nt
```

For the fixed configuration:

``` python
assert Nr == 2
assert Nt == Nx ** 2
assert L == 4
assert RFno == 1
```

If any assertion fails, reject the sample rather than silently modifying
the tensor.

## Important issue with `test_size`

`test_size` changes the batch dimension. That is acceptable during
simulation, but it should not be considered a physical system parameter
for the learned model.

For a clean ML dataset, use a fixed batch size, for example:

``` python
test_size = 20
```

and treat every generated realization inside the batch as a channel
realization.

If `test_size` is randomized, the number of channel realizations per
dataset row changes, which complicates tensor storage and training.

## Recommended final dataset design

The cleanest configuration for the first 10K dataset is:

``` python
dmode = 1

Nr = 2
Nx = 16
Nt = 256
L = 4
RFno = 1
test_size = 20

frequency = 7–15 GHz
SNR = 0–40 dB
noise_var = randomized
scale = randomized
wavelength = c / frequency
deH = wavelength / 5
deP = wavelength / 2
stepSize1 = randomized
lopPara = randomized
```

Only physical/environmental/control parameters should vary.

Matrix dimensions should remain fixed.

## Project goal

The final dataset is intended to support:

1.  Deep-learning prediction of GP beamforming.
2.  Prediction of final `w`.
3.  Prediction of `M`.
4.  Prediction of normalized mean rate.
5.  Replacement/surrogate modeling of computationally expensive GP
    iterations.
6.  Comparison between classical GP beamforming and learned beamforming.
7.  Future 6G beamforming optimization under varying frequency, SNR,
    noise, and channel conditions.

## Current blocking issue to solve next

The current GP code has dimensional errors because the gradient
derivation is mixing:

``` text
Nr dimension
Nt dimension
RFno dimension
L dimension
```

The next implementation should derive the gradient with explicit tensor
dimensions rather than repeatedly forcing tensors to match with
`repeat()` or `expand()`.

The fixed-dimension setup above is intended to make that derivation
stable and reproducible.

## Important terminology

Do not claim that every selected numerical range is an official 6G
standard requirement.

Use wording such as:

> "6G-oriented simulation parameter range based on the IMT-2030 research
> and spectrum-development context."

The ITU IMT-2030 framework defines the overall 6G framework and
evaluation process; it does not prescribe the exact random ranges used
in this simulation. Current spectrum studies include portions of the
7--15 GHz region, particularly 7.125--8.4 GHz and 14.8--15.35 GHz in the
Indian government summary.

## Continuation instruction

When extending this project, preserve:

``` text
Nr = 2
Nt = Nx²
L = fixed
RFno = fixed
H and E identical Nt dimension
No arbitrary tensor repeat/expand for mathematical compatibility
10,000 randomized simulation samples
CSV scalar metadata
Separate tensor storage for complex beamforming outputs
6G-oriented 7–15 GHz frequency range
```

The primary objective is a physically meaningful, dimensionally
consistent dataset suitable for supervised deep-learning beamforming
research.
