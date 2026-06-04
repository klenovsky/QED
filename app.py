import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import eigh

st.set_page_config(page_title="Quantum Electrodynamics Explorer", layout="wide")

TEXT = {
    "en": {
        "app_title": "Quantum Electrodynamics Explorer",
        "subtitle": "An interactive introduction to quantized light, a two-level atom, and the Jaynes–Cummings model.",
        "language": "Language",
        "theory_title": "Theory and context",
        "theory_body": r"""
This app uses a minimal and visually accessible model of quantum electrodynamics: one **quantized cavity mode** interacting with one **two-level atom**.

### Core ingredients

For the field, the photon-number states are $|n\rangle$ with energies

$$
E_n = \hbar \omega_c \left(n + \frac{1}{2}\right).
$$

The creation and annihilation operators $a^\dagger$ and $a$ change the photon number by one.

For the atom, the two basis states are the ground state $|g\rangle$ and excited state $|e\rangle$, separated by an energy $\hbar \omega_0$.

The combined light–matter dynamics is described here with the **Jaynes–Cummings Hamiltonian**

$$
H = \hbar \omega_c\, a^\dagger a + \frac{\hbar \omega_0}{2}\sigma_z + \hbar g \left(a^\dagger \sigma_- + a \sigma_+\right),
$$

where $g$ is the coupling strength.

### What to look for

- **Quantized light:** photon-number distributions are discrete.
- **Resonant exchange:** excitation oscillates between atom and field.
- **Vacuum Rabi oscillations:** even the state $|e,0\rangle$ evolves non-trivially.
- **Collapse and revival:** with a coherent field, many nearby Rabi frequencies first dephase and later rephase.
- **Detuning:** when $\omega_0 \neq \omega_c$, energy exchange becomes less efficient.

### References

- C. Cohen-Tannoudji, J. Dupont-Roc, G. Grynberg, *Photons and Atoms*.
- M. O. Scully, M. S. Zubairy, *Quantum Optics*.
- D. F. Walls, G. J. Milburn, *Quantum Optics*.
- C. Gerry, P. Knight, *Introductory Quantum Optics*.
""",
        "tab_light": "Quantized light",
        "tab_jc": "Atom–field dynamics",
        "tab_vacuum": "Vacuum Rabi",
        "tab_revival": "Collapse & revival",
        "tab_detuning": "Detuning",
        "help_light": "How to use this panel",
        "help_light_body": """
Choose a quantum state of one cavity mode and inspect its photon-number distribution $P_n$.

- **Fock state:** exactly one photon number is occupied.
- **Coherent state:** Poisson-like distribution, closest to a classical single-mode field.
- **Thermal state:** broad distribution with stronger fluctuations.

The bar chart shows the probability of finding $n$ photons. The metrics below it summarize the mean photon number and the variance.
""",
        "help_jc": "How to use this panel",
        "help_jc_body": """
Choose the initial atomic state, the initial photon number, the coupling strength, and the detuning.

What you see:
- **Left:** populations of the excited and ground atomic states versus time.
- **Right:** the mean photon number versus time.
- **Animation:** a moving marker traces the time evolution, while the lower bar chart shows the instantaneous photon distribution in the field.

On resonance, the excitation swaps most efficiently between the atom and the field.
""",
        "help_vacuum": "How to use this panel",
        "help_vacuum_body": """
This panel fixes the initial state to $|e,0\rangle$.

Although there is no classical light in the cavity, the atom and the quantized field still exchange excitation coherently. That is the basic vacuum Rabi effect in this minimal model.
""",
        "help_revival": "How to use this panel",
        "help_revival_body": """
The atom starts excited and the field starts in a coherent state.

Because a coherent state contains many photon numbers at once, several nearby Rabi frequencies participate in the dynamics. At first they dephase, which produces a collapse of the oscillations. Later they rephase, which produces a revival.

- **Top:** atomic excitation probability.
- **Bottom:** mean photon number.
- **Animation:** the marker shows the current time during the evolution.
""",
        "help_detuning": "How to use this panel",
        "help_detuning_body": r"""
This panel scans the detuning $\Delta = \omega_0 - \omega_c$.

- **Heatmap:** excited-state population $P_e(t)$ as a function of time and detuning.
- **Line plot:** a selected detuning cut through the heatmap.

Near resonance the atom–field exchange is strongest. Far from resonance it becomes weaker and faster in phase.
""",
        "reset": "Reset section",
        "nmax": "Maximum photon number",
        "state_type": "Field state",
        "fock": "Fock state",
        "coherent": "Coherent state",
        "thermal": "Thermal-like state",
        "fock_n": "Photon number n",
        "alpha": "Coherent amplitude |α|",
        "nbar": "Mean photon number",
        "distribution": "Photon-number distribution",
        "mean_n": "Mean photon number",
        "var_n": "Variance",
        "peak_n": "Most likely n",
        "initial_atom": "Initial atomic state",
        "excited": "Excited |e⟩",
        "ground": "Ground |g⟩",
        "initial_photons": "Initial photon number n₀",
        "coupling": "Coupling g / ωc",
        "detuning": "Detuning Δ / ωc",
        "tmax": "Maximum time in units of 1/ωc",
        "frames": "Animation frames",
        "time": "Time",
        "excited_pop": "Excited population",
        "ground_pop": "Ground population",
        "mean_photons": "Mean photons",
        "inst_field_dist": "Instantaneous field photon distribution",
        "selected_detuning": "Selected detuning cut",
        "detuning_range": "Maximum |Δ| / ωc for scan",
        "vacuum_note": "Initial state fixed to |e,0⟩.",
        "realizations_note": "All calculations are done in dimensionless units with ħ = 1 and ωc = 1.",
        "footer": "Numerics: vectorized NumPy/SciPy implementation with dense diagonalization of the Jaynes–Cummings Hamiltonian.",
    },
    "cs": {
        "app_title": "Průzkumník kvantové elektrodynamiky",
        "subtitle": "Interaktivní úvod do kvantovaného světla, dvouhladinového atomu a Jaynes–Cummingsova modelu.",
        "language": "Jazyk",
        "theory_title": "Teorie a kontext",
        "theory_body": r"""
Tato aplikace používá minimální a vizuálně dobře uchopitelný model kvantové elektrodynamiky: jeden **kvantovaný mód dutiny** interagující s jedním **dvouhladinovým atomem**.

### Základní stavební prvky

Pro pole jsou stavy s pevným počtem fotonů $|n\rangle$ a jejich energie

$$
E_n = \hbar \omega_c \left(n + \frac{1}{2}\right).
$$

Operátory $a^\dagger$ a $a$ mění počet fotonů o jedničku.

Pro atom používáme dva stavy: základní stav $|g\rangle$ a excitovaný stav $|e\rangle$, oddělené energií $\hbar \omega_0$.

Spojená dynamika pole a atomu je zde popsána **Jaynes–Cummingsovým Hamiltoniánem**

$$
H = \hbar \omega_c\, a^\dagger a + \frac{\hbar \omega_0}{2}\sigma_z + \hbar g \left(a^\dagger \sigma_- + a \sigma_+\right),
$$

kde $g$ je síla vazby.

### Na co se zaměřit

- **Kvantované světlo:** rozdělení počtu fotonů je diskrétní.
- **Rezonanční výměna:** excitace osciluje mezi atomem a polem.
- **Vakuové Rabiho oscilace:** i stav $|e,0\rangle$ má netriviální vývoj.
- **Kolaps a revival:** koherentní pole vede nejprve k rozfázování a poté k opětovnému složení oscilací.
- **Detuning:** když $\omega_0 \neq \omega_c$, přenos energie je méně účinný.

### Reference

- C. Cohen-Tannoudji, J. Dupont-Roc, G. Grynberg, *Photons and Atoms*.
- M. O. Scully, M. S. Zubairy, *Quantum Optics*.
- D. F. Walls, G. J. Milburn, *Quantum Optics*.
- C. Gerry, P. Knight, *Introductory Quantum Optics*.
""",
        "tab_light": "Kvantované světlo",
        "tab_jc": "Dynamika atom–pole",
        "tab_vacuum": "Vakuové Rabiho oscilace",
        "tab_revival": "Kolaps a revival",
        "tab_detuning": "Detuning",
        "help_light": "Jak tento panel používat",
        "help_light_body": """
Vyber kvantový stav jednoho módu dutiny a prohlédni si rozdělení pravděpodobnosti $P_n$ pro počet fotonů.

- **Fockův stav:** obsazen je právě jeden počet fotonů.
- **Koherentní stav:** Poissonovské rozdělení, nejbližší klasickému jednomódovému poli.
- **Tepelný stav:** široké rozdělení se silnějšími fluktuacemi.

Sloupcový graf ukazuje pravděpodobnost výskytu $n$ fotonů. Pod ním jsou shrnuty střední hodnota a variance.
""",
        "help_jc": "Jak tento panel používat",
        "help_jc_body": """
Zvol počáteční stav atomu, počáteční počet fotonů, sílu vazby a detuning.

Co je zobrazeno:
- **Vlevo:** populace excitovaného a základního stavu atomu v čase.
- **Vpravo:** střední počet fotonů v čase.
- **Animace:** pohyblivý bod ukazuje aktuální čas a spodní sloupcový graf ukazuje okamžité rozdělení fotonů v poli.

Na rezonanci se excitace mezi atomem a polem přelévá nejúčinněji.
""",
        "help_vacuum": "Jak tento panel používat",
        "help_vacuum_body": """
Tento panel fixuje počáteční stav na $|e,0\rangle$.

I když v dutině není žádné klasické světlo, atom a kvantované pole si stále koherentně vyměňují excitaci. To je základní vakuový Rabiho jev v tomto minimálním modelu.
""",
        "help_revival": "Jak tento panel používat",
        "help_revival_body": """
Atom začíná v excitovaném stavu a pole v koherentním stavu.

Protože koherentní stav obsahuje současně více počtů fotonů, účastní se dynamiky více blízkých Rabiho frekvencí. Nejprve se rozfázují, což vede ke kolapsu oscilací. Později se znovu složí a objeví se revival.

- **Nahoře:** pravděpodobnost excitovaného stavu atomu.
- **Dole:** střední počet fotonů.
- **Animace:** bod ukazuje aktuální čas v průběhu vývoje.
""",
        "help_detuning": "Jak tento panel používat",
        "help_detuning_body": r"""
Tento panel skenuje detuning $\Delta = \omega_0 - \omega_c$.

- **Heatmapa:** populace excitovaného stavu $P_e(t)$ jako funkce času a detuningu.
- **Čárový graf:** vybraný řez heatmapou pro konkrétní detuning.

V blízkosti rezonance je výměna mezi atomem a polem nejsilnější. Daleko od rezonance slábne a mění charakter oscilací.
""",
        "reset": "Reset sekce",
        "nmax": "Maximální počet fotonů",
        "state_type": "Stav pole",
        "fock": "Fockův stav",
        "coherent": "Koherentní stav",
        "thermal": "Tepelný stav",
        "fock_n": "Počet fotonů n",
        "alpha": "Koherentní amplituda |α|",
        "nbar": "Střední počet fotonů",
        "distribution": "Rozdělení počtu fotonů",
        "mean_n": "Střední počet fotonů",
        "var_n": "Variance",
        "peak_n": "Nejpravděpodobnější n",
        "initial_atom": "Počáteční stav atomu",
        "excited": "Excitovaný |e⟩",
        "ground": "Základní |g⟩",
        "initial_photons": "Počáteční počet fotonů n₀",
        "coupling": "Vazba g / ωc",
        "detuning": "Detuning Δ / ωc",
        "tmax": "Maximální čas v jednotkách 1/ωc",
        "frames": "Počet snímků animace",
        "time": "Čas",
        "excited_pop": "Populace excitovaného stavu",
        "ground_pop": "Populace základního stavu",
        "mean_photons": "Střední počet fotonů",
        "inst_field_dist": "Okamžité rozdělení fotonů v poli",
        "selected_detuning": "Vybraný řez detuningem",
        "detuning_range": "Maximální |Δ| / ωc pro scan",
        "vacuum_note": "Počáteční stav je fixován na |e,0⟩.",
        "realizations_note": "Všechny výpočty jsou v bezrozměrných jednotkách s ħ = 1 a ωc = 1.",
        "footer": "Numerika: vektorizovaná implementace v NumPy/SciPy s hustou diagonalizací Jaynes–Cummingsova Hamiltoniánu.",
    },
}

DEFAULTS = {
    "light": {"state_type": "coherent", "nmax": 20, "fock_n": 3, "alpha": 2.0, "nbar": 4.0},
    "jc": {"atom": "e", "n0": 2, "nmax": 12, "g": 0.08, "delta": 0.0, "tmax": 120.0, "frames": 48},
    "vacuum": {"nmax": 8, "g": 0.10, "delta": 0.0, "tmax": 120.0, "frames": 48},
    "revival": {"nmax": 30, "alpha": 4.0, "g": 0.05, "delta": 0.0, "tmax": 420.0, "frames": 64},
    "detuning": {"nmax": 10, "n0": 1, "g": 0.08, "dmax": 0.40, "tmax": 160.0, "ndelta": 61},
}


def tr(key: str) -> str:
    return TEXT[st.session_state.app_lang][key]


def init_language():
    if "app_lang" not in st.session_state:
        st.session_state.app_lang = "en"


def set_defaults(prefix: str):
    for k, v in DEFAULTS[prefix].items():
        st.session_state[f"{prefix}_{k}"] = v


def ensure_defaults():
    for prefix in DEFAULTS:
        for k, v in DEFAULTS[prefix].items():
            st.session_state.setdefault(f"{prefix}_{k}", v)


def reset_section(prefix: str):
    set_defaults(prefix)
    st.rerun()


@st.cache_resource
def field_operators(nmax: int):
    dim = nmax + 1
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    adag = a.conj().T
    num = adag @ a
    ident = np.eye(dim, dtype=complex)
    return a, adag, num, ident


@st.cache_resource
def jc_eigendecomposition(nmax: int, g: float, delta: float):
    a, adag, num, ident_f = field_operators(nmax)
    wc = 1.0
    w0 = wc + delta
    sigma_z = np.array([[-1, 0], [0, 1]], dtype=complex)
    sigma_p = np.array([[0, 0], [1, 0]], dtype=complex)
    sigma_m = np.array([[0, 1], [0, 0]], dtype=complex)
    ident_a = np.eye(2, dtype=complex)

    H = (
        wc * np.kron(num, ident_a)
        + 0.5 * w0 * np.kron(ident_f, sigma_z)
        + g * (np.kron(adag, sigma_m) + np.kron(a, sigma_p))
    )
    evals, evecs = eigh(H)
    return evals, evecs


def basis_state(nmax: int, atom: str, n0: int):
    vec = np.zeros((2 * (nmax + 1),), dtype=complex)
    atom_idx = 1 if atom == "e" else 0
    vec[2 * n0 + atom_idx] = 1.0
    return vec


def coherent_field(nmax: int, alpha: float):
    n = np.arange(nmax + 1)
    factorials = np.array([math.factorial(int(k)) for k in n], dtype=float)
    coeffs = np.exp(-0.5 * alpha**2) * (alpha**n) / np.sqrt(factorials)
    coeffs /= np.linalg.norm(coeffs)
    return coeffs.astype(complex)


def thermal_distribution(nmax: int, nbar: float):
    n = np.arange(nmax + 1, dtype=float)
    p = (nbar**n) / ((1.0 + nbar) ** (n + 1.0))
    p /= p.sum()
    return p


def field_distribution(state_type: str, nmax: int, fock_n: int, alpha: float, nbar: float):
    n = np.arange(nmax + 1)
    if state_type == "fock":
        p = np.zeros_like(n, dtype=float)
        p[min(fock_n, nmax)] = 1.0
    elif state_type == "coherent":
        coeffs = coherent_field(nmax, alpha)
        p = np.abs(coeffs) ** 2
    else:
        p = thermal_distribution(nmax, nbar)
    p /= p.sum()
    return n, p


def initial_product_state(nmax: int, atom: str, field_coeffs: np.ndarray):
    vec = np.zeros((2 * (nmax + 1),), dtype=complex)
    atom_idx = 1 if atom == "e" else 0
    vec[atom_idx::2] = field_coeffs
    return vec


def evolve_states(nmax: int, psi0: np.ndarray, g: float, delta: float, times: np.ndarray):
    evals, evecs = jc_eigendecomposition(nmax, float(g), float(delta))
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times, evals))
    states = (phases * coeffs[None, :]) @ evecs.T
    return states.reshape(len(times), nmax + 1, 2)


def observables_from_states(states: np.ndarray):
    probs = np.abs(states) ** 2
    pe = probs[:, :, 1].sum(axis=1)
    pg = probs[:, :, 0].sum(axis=1)
    n = np.arange(states.shape[1])[None, :]
    mean_n = (probs.sum(axis=2) * n).sum(axis=1)
    field_dist = probs.sum(axis=2)
    return pe, pg, mean_n, field_dist


def animated_jc_figure(times, pe, pg, mean_n, field_dist):
    max_n = field_dist.shape[1] - 1
    n = np.arange(max_n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=pe, mode="lines", name=tr("excited_pop"), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=times, y=pg, mode="lines", name=tr("ground_pop"), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=times, y=mean_n, mode="lines", name=tr("mean_photons"), xaxis="x2", yaxis="y2"))
    fig.add_trace(go.Scatter(x=[times[0]], y=[pe[0]], mode="markers", marker=dict(size=11), showlegend=False, xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=[times[0]], y=[mean_n[0]], mode="markers", marker=dict(size=11), showlegend=False, xaxis="x2", yaxis="y2"))
    fig.add_trace(go.Bar(x=n, y=field_dist[0], name=tr("inst_field_dist"), xaxis="x3", yaxis="y3"))

    frames = []
    for k in range(len(times)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=times, y=pe),
                    go.Scatter(x=times, y=pg),
                    go.Scatter(x=times, y=mean_n),
                    go.Scatter(x=[times[k]], y=[pe[k]]),
                    go.Scatter(x=[times[k]], y=[mean_n[k]]),
                    go.Bar(x=n, y=field_dist[k]),
                ],
                name=str(k),
            )
        )

    fig.frames = frames
    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=40, b=20),
        grid=dict(rows=2, columns=2, pattern="independent"),
        xaxis=dict(title=tr("time")),
        yaxis=dict(title="P"),
        xaxis2=dict(title=tr("time")),
        yaxis2=dict(title=tr("mean_photons")),
        xaxis3=dict(title="n"),
        yaxis3=dict(title="P(n)"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02,
                y=1.07,
            )
        ],
    )
    return fig


def animated_marker_figure(times, y, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=y, mode="lines", name=y_label))
    fig.add_trace(go.Scatter(x=[times[0]], y=[y[0]], mode="markers", marker=dict(size=11), showlegend=False))
    fig.frames = [
        go.Frame(data=[go.Scatter(x=times, y=y), go.Scatter(x=[times[k]], y=[y[k]])], name=str(k))
        for k in range(len(times))
    ]
    fig.update_layout(
        height=420,
        xaxis_title=tr("time"),
        yaxis_title=y_label,
        margin=dict(l=20, r=20, t=40, b=20),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02,
                y=1.08,
            )
        ],
    )
    return fig


def detuning_scan(nmax, n0, g, dmax, tmax, ndelta):
    deltas = np.linspace(-dmax, dmax, ndelta)
    times = np.linspace(0.0, tmax, 220)
    psi0 = basis_state(nmax, "e", n0)
    pe_all = np.empty((len(deltas), len(times)))
    for i, delta in enumerate(deltas):
        states = evolve_states(nmax, psi0, g, float(delta), times)
        pe, _, _, _ = observables_from_states(states)
        pe_all[i] = pe
    return deltas, times, pe_all


def top_language_selector():
    lang = st.radio(
        tr("language"),
        options=["en", "cs"],
        format_func=lambda x: "English" if x == "en" else "Čeština",
        horizontal=True,
        index=0 if st.session_state.app_lang == "en" else 1,
    )
    st.session_state.app_lang = lang


def main():
    init_language()
    ensure_defaults()
    top_language_selector()

    st.title(tr("app_title"))
    st.caption(tr("subtitle"))

    with st.expander(tr("theory_title"), expanded=False):
        st.markdown(TEXT[st.session_state.app_lang]["theory_body"])

    tabs = st.tabs([tr("tab_light"), tr("tab_jc"), tr("tab_vacuum"), tr("tab_revival"), tr("tab_detuning")])

    with tabs[0]:
        with st.expander(tr("help_light"), expanded=False):
            st.markdown(tr("help_light_body"))
        c1, c2, c3 = st.columns([1.0, 1.0, 0.6])
        with c1:
            st.selectbox(tr("state_type"), options=["fock", "coherent", "thermal"], format_func=lambda x: tr(x), key="light_state_type")
            st.slider(tr("nmax"), 6, 60, key="light_nmax")
            if st.session_state.light_state_type == "fock":
                st.slider(tr("fock_n"), 0, st.session_state.light_nmax, key="light_fock_n")
            elif st.session_state.light_state_type == "coherent":
                st.slider(tr("alpha"), 0.2, 6.0, step=0.1, key="light_alpha")
            else:
                st.slider(tr("nbar"), 0.2, 10.0, step=0.1, key="light_nbar")
        with c2:
            n, p = field_distribution(
                st.session_state.light_state_type,
                st.session_state.light_nmax,
                st.session_state.light_fock_n,
                st.session_state.light_alpha,
                st.session_state.light_nbar,
            )
            fig = go.Figure([go.Bar(x=n, y=p)])
            fig.update_layout(height=420, xaxis_title="n", yaxis_title="P(n)", title=tr("distribution"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            mean_n = float((n * p).sum())
            var_n = float(((n**2) * p).sum() - mean_n**2)
            st.metric(tr("mean_n"), f"{mean_n:.3f}")
            st.metric(tr("var_n"), f"{var_n:.3f}")
            st.metric(tr("peak_n"), str(int(n[np.argmax(p)])))
            st.button(tr("reset"), key="reset_light", on_click=reset_section, args=("light",))
        st.caption(tr("realizations_note"))

    with tabs[1]:
        with st.expander(tr("help_jc"), expanded=False):
            st.markdown(tr("help_jc_body"))
        cols = st.columns([1.0, 1.0, 0.7])
        with cols[0]:
            st.selectbox(tr("initial_atom"), options=["e", "g"], format_func=lambda x: tr("excited") if x == "e" else tr("ground"), key="jc_atom")
            st.slider(tr("initial_photons"), 0, 8, key="jc_n0")
            st.slider(tr("nmax"), 4, 24, key="jc_nmax")
        with cols[1]:
            st.slider(tr("coupling"), 0.01, 0.25, step=0.01, key="jc_g")
            st.slider(tr("detuning"), -0.40, 0.40, step=0.01, key="jc_delta")
            st.slider(tr("tmax"), 20.0, 240.0, step=5.0, key="jc_tmax")
            st.slider(tr("frames"), 20, 120, key="jc_frames")
        with cols[2]:
            st.button(tr("reset"), key="reset_jc", on_click=reset_section, args=("jc",))

        times = np.linspace(0.0, st.session_state.jc_tmax, int(st.session_state.jc_frames))
        psi0 = basis_state(st.session_state.jc_nmax, st.session_state.jc_atom, min(st.session_state.jc_n0, st.session_state.jc_nmax))
        states = evolve_states(st.session_state.jc_nmax, psi0, st.session_state.jc_g, st.session_state.jc_delta, times)
        pe, pg, mean_n, field_dist = observables_from_states(states)

        cplot1, cplot2 = st.columns(2)
        with cplot1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=times, y=pe, mode="lines", name=tr("excited_pop")))
            fig1.add_trace(go.Scatter(x=times, y=pg, mode="lines", name=tr("ground_pop")))
            fig1.update_layout(height=380, xaxis_title=tr("time"), yaxis_title="P", margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig1, use_container_width=True)
        with cplot2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=times, y=mean_n, mode="lines", name=tr("mean_photons")))
            fig2.update_layout(height=380, xaxis_title=tr("time"), yaxis_title=tr("mean_photons"), margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(animated_jc_figure(times, pe, pg, mean_n, field_dist), use_container_width=True)

    with tabs[2]:
        with st.expander(tr("help_vacuum"), expanded=False):
            st.markdown(tr("help_vacuum_body"))
        cols = st.columns([1.0, 1.0, 0.7])
        with cols[0]:
            st.slider(tr("nmax"), 2, 20, key="vacuum_nmax")
            st.slider(tr("coupling"), 0.01, 0.25, step=0.01, key="vacuum_g")
        with cols[1]:
            st.slider(tr("detuning"), -0.40, 0.40, step=0.01, key="vacuum_delta")
            st.slider(tr("tmax"), 20.0, 240.0, step=5.0, key="vacuum_tmax")
            st.slider(tr("frames"), 20, 120, key="vacuum_frames")
        with cols[2]:
            st.info(tr("vacuum_note"))
            st.button(tr("reset"), key="reset_vacuum", on_click=reset_section, args=("vacuum",))

        times = np.linspace(0.0, st.session_state.vacuum_tmax, int(st.session_state.vacuum_frames))
        psi0 = basis_state(st.session_state.vacuum_nmax, "e", 0)
        states = evolve_states(st.session_state.vacuum_nmax, psi0, st.session_state.vacuum_g, st.session_state.vacuum_delta, times)
        pe, pg, mean_n, field_dist = observables_from_states(states)

        cplot1, cplot2 = st.columns(2)
        with cplot1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=times, y=pe, mode="lines", name=tr("excited_pop")))
            fig1.add_trace(go.Scatter(x=times, y=pg, mode="lines", name=tr("ground_pop")))
            fig1.update_layout(height=380, xaxis_title=tr("time"), yaxis_title="P", margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig1, use_container_width=True)
        with cplot2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=times, y=mean_n, mode="lines", name=tr("mean_photons")))
            fig2.update_layout(height=380, xaxis_title=tr("time"), yaxis_title=tr("mean_photons"), margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(animated_jc_figure(times, pe, pg, mean_n, field_dist), use_container_width=True)

    with tabs[3]:
        with st.expander(tr("help_revival"), expanded=False):
            st.markdown(tr("help_revival_body"))
        cols = st.columns([1.0, 1.0, 0.7])
        with cols[0]:
            st.slider(tr("nmax"), 8, 50, key="revival_nmax")
            st.slider(tr("alpha"), 0.5, 6.5, step=0.1, key="revival_alpha")
        with cols[1]:
            st.slider(tr("coupling"), 0.01, 0.20, step=0.01, key="revival_g")
            st.slider(tr("detuning"), -0.30, 0.30, step=0.01, key="revival_delta")
            st.slider(tr("tmax"), 80.0, 700.0, step=10.0, key="revival_tmax")
            st.slider(tr("frames"), 30, 180, key="revival_frames")
        with cols[2]:
            st.button(tr("reset"), key="reset_revival", on_click=reset_section, args=("revival",))

        field = coherent_field(st.session_state.revival_nmax, st.session_state.revival_alpha)
        psi0 = initial_product_state(st.session_state.revival_nmax, "e", field)
        times = np.linspace(0.0, st.session_state.revival_tmax, int(st.session_state.revival_frames))
        states = evolve_states(st.session_state.revival_nmax, psi0, st.session_state.revival_g, st.session_state.revival_delta, times)
        pe, pg, mean_n, field_dist = observables_from_states(states)

        cplot1, cplot2 = st.columns(2)
        with cplot1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=times, y=pe, mode="lines", name=tr("excited_pop")))
            fig1.update_layout(height=380, xaxis_title=tr("time"), yaxis_title=tr("excited_pop"), margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        with cplot2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=times, y=mean_n, mode="lines", name=tr("mean_photons")))
            fig2.update_layout(height=380, xaxis_title=tr("time"), yaxis_title=tr("mean_photons"), margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(animated_marker_figure(times, pe, tr("excited_pop")), use_container_width=True)

    with tabs[4]:
        with st.expander(tr("help_detuning"), expanded=False):
            st.markdown(tr("help_detuning_body"))
        cols = st.columns([1.0, 1.0, 0.7])
        with cols[0]:
            st.slider(tr("nmax"), 4, 16, key="detuning_nmax")
            st.slider(tr("initial_photons"), 0, 4, key="detuning_n0")
            st.slider(tr("coupling"), 0.01, 0.25, step=0.01, key="detuning_g")
        with cols[1]:
            st.slider(tr("detuning_range"), 0.10, 0.80, step=0.02, key="detuning_dmax")
            st.slider(tr("tmax"), 40.0, 260.0, step=5.0, key="detuning_tmax")
            st.slider("NΔ", 21, 121, step=10, key="detuning_ndelta")
        with cols[2]:
            st.button(tr("reset"), key="reset_detuning", on_click=reset_section, args=("detuning",))

        deltas, times, pe_map = detuning_scan(
            st.session_state.detuning_nmax,
            st.session_state.detuning_n0,
            st.session_state.detuning_g,
            st.session_state.detuning_dmax,
            st.session_state.detuning_tmax,
            st.session_state.detuning_ndelta,
        )
        fig = go.Figure(data=go.Heatmap(x=times, y=deltas, z=pe_map, colorbar_title=tr("excited_pop")))
        fig.update_layout(height=420, xaxis_title=tr("time"), yaxis_title=tr("detuning"))
        st.plotly_chart(fig, use_container_width=True)
        delta_sel = st.slider(tr("detuning"), float(deltas.min()), float(deltas.max()), 0.0, step=0.01)
        idx = int(np.argmin(np.abs(deltas - delta_sel)))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times, y=pe_map[idx], mode="lines", name=tr("selected_detuning")))
        fig2.update_layout(height=360, xaxis_title=tr("time"), yaxis_title=tr("excited_pop"), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(tr("footer"))


if __name__ == "__main__":
    main()
