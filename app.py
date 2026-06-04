import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import eig, eigh

st.set_page_config(page_title="Quantum Electrodynamics Explorer", layout="wide")


TEXT = {
    "en": {
        "app_title": "Quantum Electrodynamics Explorer",
        "subtitle": "An interactive view of cavity QED: quantized light, a two-level atom, and Jaynes–Cummings dynamics in the broader context of quantum mechanics and modern physics.",
        "language": "Language",
        "theory_title": "Theory, context, and references",
        "theory_body": r"""
This app sits at the boundary between **quantum mechanics**, **quantized radiation**, and **light–matter interaction**. In ordinary introductory quantum mechanics one first learns discrete levels, superposition, and two-level systems. Quantum electrodynamics extends this logic to the electromagnetic field itself: instead of treating light only as a classical wave, one quantizes the field and obtains photon-number states, creation and annihilation operators, and genuinely quantum atom–field exchange. In practice, one of the cleanest and most useful settings is **cavity QED**, where a single field mode interacts strongly with a single effective two-level system. That controlled setting is the focus of this simulator [1–6].

### Minimal model used here

For the field, the photon-number states are $|n\rangle$ with energies

$$
E_n = \hbar \omega_c \left(n + \frac{1}{2}\right).
$$

The operators $a^\dagger$ and $a$ raise and lower the photon number by one. For the atom, the basis states are the ground state $|g\rangle$ and excited state $|e\rangle$, separated by $\hbar \omega_0$.

The coupled dynamics is described here by the **Jaynes–Cummings Hamiltonian**

$$
H = \hbar \omega_c\, a^\dagger a + \frac{\hbar \omega_0}{2}\sigma_z + \hbar g \left(a^\dagger \sigma_- + a \sigma_+\right),
$$

with coupling strength $g$. This model is historically central because it makes explicit how one quantum of excitation can be coherently exchanged between matter and a single quantized radiation mode [1]. In modern language, it is one of the standard model Hamiltonians of cavity QED and quantum optics [3–6].

### Why this is physically important

The simulator brings together several ideas that often appear separately in courses:

- **Quantized light:** the field has discrete excitation sectors labelled by photon number, not just a continuous classical amplitude [2–4].
- **Two-level dynamics:** a two-state atom is the simplest nontrivial quantum system and naturally leads to Bloch-sphere intuition [3,4].
- **Coherent exchange:** on or near resonance, excitation oscillates between atom and field rather than decaying irreversibly in the ideal closed model [1,3–6].
- **Vacuum effects:** even the state $|e,0\rangle$ evolves nontrivially because the vacuum of the quantized field is not equivalent to “nothing happening” in the coupled quantum problem [3,5,6].
- **Collapse and revival:** if the field begins in a coherent state, many nearby Rabi frequencies participate, producing dephasing and later rephasing of the atomic signal [3,4,7].

### What is shown here — and what is not

This app is intentionally a **minimal closed-system model**. It uses:
- a single cavity mode,
- one effective two-level atom,
- the rotating-wave form of the interaction,
- a truncated photon basis for numerics,
- dimensionless units with $\hbar=1$ and $\omega_c=1$.

That makes the central physics transparent, but it also means this is **not** a full relativistic QED calculation. The first panels deliberately isolate the coherent, nearly textbook Jaynes–Cummings limit. The final open-system panel then adds a simple **Lindblad master-equation** description of cavity loss, atomic relaxation, and dephasing, which is the standard next step when one wants to connect ideal cavity-QED motion to laboratory reality [8–10]. Even that extension is still intentionally minimal: there is no multimode continuum, no external driving, no ultrastrong-coupling physics, and no renormalization. The goal is clarity: to isolate the clean coherent structures first and then show how dissipation reshapes them [3–10].

### Selected references

[1] E. T. Jaynes and F. W. Cummings, *Comparison of quantum and semiclassical radiation theories with application to the beam maser*, Proc. IEEE **51**, 89–109 (1963).  
[2] C. Cohen-Tannoudji, J. Dupont-Roc, and G. Grynberg, *Photons and Atoms: Introduction to Quantum Electrodynamics* (Wiley, 1989).  
[3] M. O. Scully and M. S. Zubairy, *Quantum Optics* (Cambridge University Press, 1997).  
[4] C. C. Gerry and P. L. Knight, *Introductory Quantum Optics*, 2nd ed. (Cambridge University Press, 2023).  
[5] J.-M. Raimond, M. Brune, and S. Haroche, *Manipulating quantum entanglement with atoms and photons in a cavity*, Rev. Mod. Phys. **73**, 565–582 (2001).  
[6] S. Haroche, *Controlling photons in a box and exploring the quantum to classical boundary*, Rev. Mod. Phys. **85**, 1083–1102 (2013).  
[7] J. H. Eberly, N. B. Narozhny, and J. J. Sánchez-Mondragón, *Periodic spontaneous collapse and revival in a simple quantum model*, Phys. Rev. Lett. **44**, 1323–1326 (1980).  
[8] H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems* (Oxford University Press, 2007).  
[9] H. J. Carmichael, *Statistical Methods in Quantum Optics 1: Master Equations and Fokker–Planck Equations* (Springer, 1999).  
[10] F. Campaioli, F. A. Pollock, and S. Vinjanampathy, *Quantum Master Equations: Tips and Tricks for Quantum Optics, Quantum Computing, and Beyond*, PRX Quantum **5**, 020202 (2024).  
[8] H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems* (Oxford University Press, 2007).  
[9] H. J. Carmichael, *Statistical Methods in Quantum Optics 1: Master Equations and Fokker–Planck Equations* (Springer, 1999).  
[10] F. Campaioli, F. A. Pollock, and S. Vinjanampathy, *Quantum Master Equations: Tips and Tricks for Quantum Optics, Quantum Computing, and Beyond*, PRX Quantum **5**, 020202 (2024).
""",
        "tab_light": "Quantized light",
        "tab_jc": "Atom–field dynamics",
        "tab_vacuum": "Vacuum Rabi",
        "tab_revival": "Collapse & revival",
        "tab_open": "Open system: damping and decoherence",
        "tab_detuning": "Detuning",
        "help_light": "How to use this panel",
        "help_light_body": r"""
This panel focuses only on the **field mode**. It is the cleanest place to compare standard quantum states of light before coupling them to the atom [2–4].

### What the controls do
- **Field state** selects the state family.
- **Maximum photon number** sets the truncation of the photon basis used in the plot.
- The last control sets the parameter of the chosen family: exact photon number $n$ for a Fock state, coherent amplitude $|\alpha|$ for a coherent state, or mean photon number $\bar n$ for a thermal-like state.

### What the graphics show
- **Bar chart:** the photon-number distribution $P_n$, i.e. the probability of finding exactly $n$ photons in the mode.
- **Metrics:** the mean photon number, the variance, and the most likely photon number.

### How to read the results
- **Fock state:** one bar is occupied. The photon number is sharp, so the variance is minimal for that fixed $n$ [2–4].
- **Coherent state:** the distribution is approximately Poissonian. Mean and variance are of comparable size, which is one reason coherent states are often regarded as the quantum states closest to a classical single-mode field [3,4].
- **Thermal-like state:** the distribution is broader and more strongly fluctuating. The field is much noisier than in a coherent state with the same mean occupation [3,4].

A useful comparison is to keep the mean photon number roughly fixed and switch between coherent and thermal-like states. The shapes of $P_n$ then immediately show how photon statistics encode physically different field states [3,4].

References: [2–4].
""",
        "help_jc": "How to use this panel",
        "help_jc_body": r"""
This is the main Jaynes–Cummings panel. It shows how one effective atomic transition exchanges excitation with one quantized cavity mode [1,3–6].

### What the controls do
- **Initial atomic state** chooses whether the atom starts in $|e\rangle$ or $|g\rangle$.
- **Initial photon number** sets the Fock-state occupation of the field at $t=0$.
- **Coupling** sets $g/\omega_c$.
- **Detuning** sets $\Delta/\omega_c = (\omega_0-\omega_c)/\omega_c$.
- **Maximum time** and **animation frames** set the displayed time window and animation resolution.

### What each graph shows
- **Left line plot:** atomic populations $P_e(t)$ and $P_g(t)$. These tell you how likely the atom is to be found excited or in the ground state.
- **Right line plot:** mean photon number $\langle n(t)\rangle$ in the field mode.
- **Animated lower panel:** the same time traces with a moving marker, plus the **instantaneous photon-number distribution** of the field.
- **Bloch sphere panel:** the reduced atomic state visualized as a point with Bloch coordinates $(x,y,z)$. Pure atomic states lie on the sphere; mixed reduced states lie inside it.

### How to interpret the dynamics
In the closed Jaynes–Cummings model, total excitation is redistributed coherently between the atom and the field. On resonance, the oscillatory exchange is strongest. Off resonance, transfer is weaker because the atom and cavity are no longer optimally matched in energy [1,3–6].

The Bloch-sphere view is especially useful: if the atom becomes entangled with the field, the **reduced atomic state** is generally no longer pure, so the trajectory moves inside the sphere rather than staying on its surface [3,4]. This makes the relation between population dynamics and coherence visually explicit.

References: [1], [3]–[6].
""",
        "help_vacuum": "How to use this panel",
        "help_vacuum_body": r"""
This panel fixes the initial state to $|e,0\rangle$: the atom starts excited and the cavity mode starts in the vacuum state.

### What each graph shows
- **Left line plot:** excited- and ground-state populations of the atom.
- **Right line plot:** mean photon number in the cavity.
- **Animated panel:** a moving marker in time together with the instantaneous field photon distribution.
- **Bloch sphere panel:** the reduced atomic state during the evolution.

### Why this panel matters
The point of this panel is conceptual. There is **no classical drive field** in the cavity, yet the coupled quantum system still evolves. In the Jaynes–Cummings model, the atomic excitation can be coherently exchanged with the quantized cavity mode, producing the basic vacuum Rabi oscillation [3–6].

This does not mean that “vacuum is full of real photons” in a naive classical sense. Rather, the quantized field has its own allowed states and operators, and the coupled atom–field Hamiltonian generates nontrivial dynamics even when the field starts in the vacuum number state [3,5,6].

References: [3]–[6].
""",
        "help_revival": "How to use this panel",
        "help_revival_body": r"""
Here the atom starts in $|e\rangle$ while the field starts in a **coherent state**. This is the panel where the celebrated **collapse and revival** phenomenon appears most clearly [3,4,7].

### What each graph shows
- **Left line plot:** excited-state probability $P_e(t)$ of the atom.
- **Right line plot:** mean photon number of the cavity mode.
- **Animated panel:** a moving marker along the atomic excitation curve.
- **Bloch sphere panel:** the reduced atomic state during the evolution.

### How to read the physics
A coherent field contains many photon-number components at once. In the Jaynes–Cummings model, different photon numbers couple with slightly different effective Rabi frequencies. At early times these contributions are still phase-aligned and the atomic signal oscillates clearly. Later they dephase, and the oscillation envelope appears to collapse. At still later times they rephase and a revival emerges [3,4,7].

This panel is a good reminder that the collapse is **not** dissipation in this ideal closed model. It is a dephasing effect inside a coherent superposition of many number sectors, and the later revival is the signature that the dynamics remained unitary [3,4,7].

References: [3], [4], [7].
""",
        "help_detuning": "How to use this panel",
        "help_detuning_body": r"""
This panel scans the atom–cavity detuning $\Delta = \omega_0 - \omega_c$ and shows how the population dynamics changes across resonance [1,3–6].

### What each graph shows
- **Heatmap:** the excited-state population $P_e(t)$ as a function of time and detuning.
- **Selected-detuning line plot:** a one-dimensional cut through the heatmap at the chosen detuning.

### How to interpret the patterns
Near $\Delta=0$, atom and cavity are resonant and exchange excitation most effectively. The heatmap therefore shows the strongest oscillation contrast around the center. As $|\Delta|$ grows, the mismatch in energy suppresses the transfer and changes the oscillation pattern [1,3–6].

The line plot is useful for connecting the global map to one concrete time trace. Try moving the detuning slider from the center outward and compare how the oscillation amplitude and apparent period change.

References: [1], [3]–[6].
""",
        "help_open": "How to use this panel",
        "help_open_body": r"""
This panel adds a simple **open-system extension** of the cavity-QED model. Instead of a pure state evolving only under the Hamiltonian, the system is described by a density matrix $\rho(t)$ obeying a Lindblad master equation [8–10].

$$
\dot\rho = -i[H,\rho] + \kappa\,\mathcal{D}[a]\rho + \gamma\,\mathcal{D}[\sigma_-]\rho + \gamma_\phi\,\mathcal{D}[\sigma_z]\rho,
$$

with $\mathcal{D}[L]\rho = L\rho L^\dagger - \tfrac12\{L^\dagger L,\rho\}$.

### What the controls do
- **Maximum photon number** sets the cavity-basis truncation.
- **Coupling** and **detuning** play the same role as in the closed Jaynes–Cummings panels.
- **Cavity loss** controls photon leakage from the cavity mode.
- **Atomic decay** transfers population from $|e\rangle$ to $|g\rangle$.
- **Pure dephasing** suppresses coherence without directly changing the atomic populations.
- **Maximum time** and **Animation frames** control the time window and temporal sampling.

### What each graph shows
- **Left line plot:** excited- and ground-state populations of the atom.
- **Right line plot:** mean cavity photon number and total purity $\mathrm{Tr}(\rho^2)$.
- **Animated multi-panel figure:** the same population and photon traces plus the instantaneous field photon-number distribution.
- **Additional purity animation:** purity as a function of time with a moving marker.
- **Bloch sphere panel:** the reduced atomic state as a trajectory inside the Bloch sphere.

### How to interpret it
In the closed model the motion is unitary and information stays inside the atom–field system. Here the environment extracts photons and coherence, so oscillations are damped, purity drops, and the Bloch-vector trajectory contracts toward the interior. Cavity loss mostly removes field excitation, atomic decay empties $|e\rangle$, and dephasing primarily kills off-diagonal coherence. Comparing this panel with the closed vacuum-Rabi and collapse–revival panels is the cleanest way to see what is lost when the environment is no longer negligible [8–10].

References: [5], [8]–[10].
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
        "cavity_loss": "Cavity loss κ / ωc",
        "atomic_decay": "Atomic decay γ / ωc",
        "dephasing": "Pure dephasing γφ / ωc",
        "purity": "Purity Tr(ρ²)",
        "inst_field_dist": "Instantaneous field photon distribution",
        "selected_detuning": "Selected detuning cut",
        "detuning_range": "Maximum |Δ| / ωc for scan",
        "vacuum_note": "Initial state fixed to $|e,0\\rangle$.",
        "realizations_note": "All calculations use dimensionless units with $\\hbar = 1$ and $\\omega_c = 1$. The photon basis is truncated at the chosen maximum photon number.",
        "footer": "Model scope: single-mode cavity-QED dynamics in Jaynes–Cummings form, with both closed-system unitary evolution and a minimal open-system Lindblad extension, implemented with vectorized NumPy/SciPy linear algebra. See the theory section for scope and references.",
    },
    "cs": {
        "app_title": "Průzkumník kvantové elektrodynamiky",
        "subtitle": "Interaktivní pohled na cavity QED: kvantované světlo, dvouhladinový atom a Jaynesův–Cummingsův model v širším kontextu kvantové mechaniky a fyziky.",
        "language": "Jazyk",
        "theory_title": "Teorie, kontext a reference",
        "theory_body": r"""
Tato aplikace leží na rozhraní **kvantové mechaniky**, **kvantovaného záření** a **interakce světla s hmotou**. V běžném úvodu do kvantové mechaniky se člověk nejprve setká s diskrétními hladinami, superpozicí a dvouúrovňovými systémy. Kvantová elektrodynamika rozšiřuje tuto logiku i na samotné elektromagnetické pole: místo čistě klasické vlny se pole kvantuje a vznikají stavy s pevným počtem fotonů, operátory vytváření a anihilace a skutečně kvantová výměna excitace mezi atomem a polem. V praxi patří mezi nejčistší a nejpoužívanější realizace **cavity QED**, kde jeden mód pole silně interaguje s jedním efektivním dvouhladinovým systémem. Právě na tuto situaci se tato simulace soustředí [1–6].

### Minimální model použitý zde

Pro pole používáme stavy s pevným počtem fotonů $|n\rangle$ a jejich energie

$$
E_n = \hbar \omega_c \left(n + \frac{1}{2}\right).
$$

Operátory $a^\dagger$ a $a$ zvyšují a snižují počet fotonů o jedničku. Pro atom používáme stavy $|g\rangle$ a $|e\rangle$, oddělené energií $\hbar \omega_0$.

Vázaná dynamika je zde popsána **Jaynesovým–Cummingsovým Hamiltoniánem**

$$
H = \hbar \omega_c\, a^\dagger a + \frac{\hbar \omega_0}{2}\sigma_z + \hbar g \left(a^\dagger \sigma_- + a \sigma_+\right),
$$

kde $g$ je síla vazby. Tento model je historicky zásadní, protože explicitně ukazuje, jak se může jeden kvant excitace koherentně přelévat mezi hmotou a jedním kvantovaným módem záření [1]. V současné terminologii jde o jeden ze standardních modelových Hamiltoniánů cavity QED a kvantové optiky [3–6].

### Proč je to fyzikálně důležité

Simulace spojuje několik myšlenek, které se ve výuce často objevují odděleně:

- **Kvantované světlo:** pole má diskrétní excitační sektory označené počtem fotonů, ne jen spojitou klasickou amplitudu [2–4].
- **Dvouúrovňová dynamika:** dvoustavový atom je nejjednodušší netriviální kvantový systém a přirozeně vede k intuici přes Blochovu kouli [3,4].
- **Koherentní výměna:** na rezonanci nebo blízko rezonance se excitace mezi atomem a polem oscilatorně přelévá místo nevratného rozpadu v ideálním uzavřeném modelu [1,3–6].
- **Vakuové efekty:** i stav $|e,0\rangle$ má netriviální vývoj, protože vakuum kvantovaného pole není vázaném systému totéž co „nic se neděje“ [3,5,6].
- **Kolaps a revival:** pokud pole začíná v koherentním stavu, účastní se dynamiky více blízkých Rabiho frekvencí, což vede k rozfázování a pozdějšímu znovusložení atomového signálu [3,4,7].

### Co tato aplikace ukazuje — a co ne

Aplikace je záměrně postavena jako **minimální uzavřený model**. Používá:
- jeden mód dutiny,
- jeden efektivní dvouhladinový atom,
- interakci ve tvaru rotating-wave approximation,
- ořezanou bázi počtu fotonů pro numeriku,
- bezrozměrné jednotky s $\hbar=1$ a $\omega_c=1$.

To zpřehledňuje základní fyziku, ale zároveň to znamená, že nejde o **plný relativistický výpočet QED**. První panely záměrně izolují koherentní, téměř učebnicový Jaynesův–Cummingsův limit. Závěrečný panel s otevřeným systémem pak přidává jednoduchý popis pomocí **Lindbladovy master equation** pro ztráty v dutině, relaxaci atomu a dephasing, což je standardní další krok při přechodu od ideální cavity QED k laboratorně realističtějšímu popisu [8–10]. I toto rozšíření je ale pořád záměrně minimální: chybí multimódové kontinuum, vnější buzení, ultrasilná vazba i renormalizace. Smyslem je nejprve izolovat čisté koherentní struktury a pak ukázat, jak je disipace přetváří [3–10].

### Vybrané reference

[1] E. T. Jaynes and F. W. Cummings, *Comparison of quantum and semiclassical radiation theories with application to the beam maser*, Proc. IEEE **51**, 89–109 (1963).  
[2] C. Cohen-Tannoudji, J. Dupont-Roc, and G. Grynberg, *Photons and Atoms: Introduction to Quantum Electrodynamics* (Wiley, 1989).  
[3] M. O. Scully and M. S. Zubairy, *Quantum Optics* (Cambridge University Press, 1997).  
[4] C. C. Gerry and P. L. Knight, *Introductory Quantum Optics*, 2nd ed. (Cambridge University Press, 2023).  
[5] J.-M. Raimond, M. Brune, and S. Haroche, *Manipulating quantum entanglement with atoms and photons in a cavity*, Rev. Mod. Phys. **73**, 565–582 (2001).  
[6] S. Haroche, *Controlling photons in a box and exploring the quantum to classical boundary*, Rev. Mod. Phys. **85**, 1083–1102 (2013).  
[7] J. H. Eberly, N. B. Narozhny, and J. J. Sánchez-Mondragón, *Periodic spontaneous collapse and revival in a simple quantum model*, Phys. Rev. Lett. **44**, 1323–1326 (1980).  
[8] H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems* (Oxford University Press, 2007).  
[9] H. J. Carmichael, *Statistical Methods in Quantum Optics 1: Master Equations and Fokker–Planck Equations* (Springer, 1999).  
[10] F. Campaioli, F. A. Pollock, and S. Vinjanampathy, *Quantum Master Equations: Tips and Tricks for Quantum Optics, Quantum Computing, and Beyond*, PRX Quantum **5**, 020202 (2024).
""",
        "tab_light": "Kvantované světlo",
        "tab_jc": "Dynamika atom–pole",
        "tab_vacuum": "Vakuové Rabiho oscilace",
        "tab_revival": "Kolaps a revival",
        "tab_open": "Otevřený systém: tlumení a dekoherence",
        "tab_detuning": "Detuning",
        "help_light": "Jak tento panel používat",
        "help_light_body": r"""
Tento panel se soustředí pouze na **mód pole**. Je to nejčistší místo pro srovnání standardních kvantových stavů světla ještě před jejich svázáním s atomem [2–4].

### Co dělají ovládací prvky
- **Stav pole** vybírá rodinu stavů.
- **Maximální počet fotonů** nastavuje ořez báze počtu fotonů používané v grafu.
- Poslední ovladač nastavuje parametr zvolené rodiny: přesný počet fotonů $n$ pro Fockův stav, koherentní amplitudu $|\alpha|$ pro koherentní stav nebo střední počet fotonů $\bar n$ pro tepelný stav.

### Co ukazují grafy
- **Sloupcový graf:** rozdělení $P_n$, tedy pravděpodobnost nalezení právě $n$ fotonů v módu.
- **Metriky:** střední počet fotonů, variance a nejpravděpodobnější hodnota $n$.

### Jak tomu rozumět
- **Fockův stav:** obsazen je jediný sloupec. Počet fotonů je ostrý, takže variance je pro dané $n$ minimální [2–4].
- **Koherentní stav:** rozdělení je přibližně Poissonovské. Střední hodnota a variance jsou srovnatelné, a právě proto jsou koherentní stavy často považovány za kvantové stavy nejbližší klasickému jednomódovému poli [3,4].
- **Tepelný stav:** rozdělení je širší a fluktuace jsou silnější. Pole je mnohem „šumovější“ než koherentní stav se stejnou střední obsazeností [3,4].

Užitečné je držet přibližně stejný střední počet fotonů a přepínat mezi koherentním a tepelným stavem. Tvar $P_n$ pak okamžitě ukáže, jak fotonová statistika rozlišuje fyzikálně odlišné stavy pole [3,4].

Reference: [2–4].
""",
        "help_jc": "Jak tento panel používat",
        "help_jc_body": r"""
Toto je hlavní panel Jaynesova–Cummingsova modelu. Ukazuje, jak si jeden efektivní atomový přechod vyměňuje excitaci s jedním kvantovaným módem dutiny [1,3–6].

### Co dělají ovládací prvky
- **Počáteční stav atomu** volí, zda atom začíná v $|e\rangle$ nebo $|g\rangle$.
- **Počáteční počet fotonů** nastavuje Fockův stav pole v čase $t=0$.
- **Vazba** nastavuje $g/\omega_c$.
- **Detuning** nastavuje $\Delta/\omega_c = (\omega_0-\omega_c)/\omega_c$.
- **Maximální čas** a **počet snímků animace** určují zobrazené časové okno a rozlišení animace.

### Co ukazuje každý graf
- **Levý čárový graf:** atomové populace $P_e(t)$ a $P_g(t)$. Ty říkají, s jakou pravděpodobností je atom excitovaný nebo v základním stavu.
- **Pravý čárový graf:** střední počet fotonů $\langle n(t)\rangle$ v módu pole.
- **Dolní animovaný panel:** stejné časové průběhy s pohyblivým bodem v čase a navíc **okamžité rozdělení počtu fotonů** v poli.
- **Panel s Blochovou koulí:** redukovaný atomový stav zobrazený bodem s Blochovými souřadnicemi $(x,y,z)$. Čisté atomové stavy leží na povrchu, smíšené redukované stavy uvnitř.

### Jak interpretovat dynamiku
V uzavřeném Jaynesově–Cummingsově modelu se celková excitace koherentně přerozděluje mezi atomem a polem. Na rezonanci je tato výměna nejsilnější. Mimo rezonanci je přenos slabší, protože atom a dutina už nejsou energeticky optimálně sladěny [1,3–6].

Pohled přes Blochovu kouli je zvlášť užitečný: pokud se atom prováže s polem, **redukovaný atomový stav** obecně přestane být čistý, a trajektorie se proto pohybuje i uvnitř koule, ne jen po povrchu [3,4]. To přímo propojuje populační dynamiku a koherenci.

Reference: [1], [3]–[6].
""",
        "help_vacuum": "Jak tento panel používat",
        "help_vacuum_body": r"""
Tento panel fixuje počáteční stav na $|e,0\rangle$: atom začíná excitovaný a mód dutiny ve vakuu.

### Co ukazuje každý graf
- **Levý čárový graf:** populace excitovaného a základního stavu atomu.
- **Pravý čárový graf:** střední počet fotonů v dutině.
- **Animovaný panel:** pohyblivý bod v čase spolu s okamžitým rozdělením počtu fotonů v poli.
- **Panel s Blochovou koulí:** redukovaný atomový stav během vývoje.

### Proč je tento panel důležitý
Smysl tohoto panelu je konceptuální. V dutině **není žádné klasické budicí pole**, a přesto se vázaný kvantový systém vyvíjí. V Jaynesově–Cummingsově modelu se atomová excitace může koherentně přelévat do kvantovaného módu pole a zpět, čímž vzniká základní vakuový Rabiho jev [3–6].

Neznamená to, že by vakuum bylo v naivním klasickém smyslu „plné reálných fotonů“. Znamená to, že kvantované pole má své vlastní stavy a operátory a že spojený Hamiltonián generuje netriviální dynamiku i tehdy, když pole začíná ve vakuovém číselném stavu [3,5,6].

Reference: [3]–[6].
""",
        "help_revival": "Jak tento panel používat",
        "help_revival_body": r"""
Zde atom začíná ve stavu $|e\rangle$ a pole v **koherentním stavu**. Právě zde se nejzřetelněji objevuje slavný jev **kolapsu a revivalu** [3,4,7].

### Co ukazuje každý graf
- **Levý čárový graf:** pravděpodobnost excitovaného stavu atomu $P_e(t)$.
- **Pravý čárový graf:** střední počet fotonů v módu dutiny.
- **Animovaný panel:** pohyblivý bod podél křivky atomové excitace.
- **Panel s Blochovou koulí:** redukovaný atomový stav během vývoje.

### Jak číst fyziku
Koherentní pole obsahuje současně více složek s různým počtem fotonů. V Jaynesově–Cummingsově modelu se různé počty fotonů pojí s mírně odlišnými efektivními Rabiho frekvencemi. Na začátku jsou tyto příspěvky ještě fázově sladěné a atomový signál dobře osciluje. Později se rozfázují, takže obálka oscilací vypadá jako kolaps. Ještě později se znovu složí a objeví se revival [3,4,7].

Tento panel je dobrou připomínkou, že kolaps v tomto ideálním uzavřeném modelu **není disipace**. Jde o dephasing uvnitř koherentní superpozice více sektorů s různým počtem fotonů a pozdější revival je známkou toho, že vývoj zůstal unitární [3,4,7].

Reference: [3], [4], [7].
""",
        "help_detuning": "Jak tento panel používat",
        "help_detuning_body": r"""
Tento panel skenuje detuning atom–dutina $\Delta = \omega_0 - \omega_c$ a ukazuje, jak se populační dynamika mění při průchodu rezonancí [1,3–6].

### Co ukazuje každý graf
- **Heatmapa:** populace excitovaného stavu $P_e(t)$ jako funkce času a detuningu.
- **Čárový graf vybraného řezu:** jednorozměrný řez heatmapou pro zvolený detuning.

### Jak interpretovat obrazce
Blízko $\Delta=0$ jsou atom a dutina na rezonanci a vyměňují si excitaci nejúčinněji. Proto heatmapa ukazuje nejsilnější kontrast oscilací uprostřed. Když roste $|\Delta|$, energetické rozladění přenos potlačuje a mění charakter oscilací [1,3–6].

Čárový graf je užitečný pro propojení globální mapy s jedním konkrétním časovým průběhem. Zkus posouvat slider detuningu od středu směrem ven a porovnat, jak se mění amplituda i zdánlivá perioda oscilací.

Reference: [1], [3]–[6].
""",
        "help_open": "Jak tento panel používat",
        "help_open_body": r"""
Tento panel přidává jednoduché **rozšíření na otevřený systém**. Místo čistého stavu, který se vyvíjí jen podle Hamiltoniánu, je systém popsán hustotní maticí $\rho(t)$ splňující Lindbladovu master equation [8–10].

$$
\dot\rho = -i[H,\rho] + \kappa\,\mathcal{D}[a]\rho + \gamma\,\mathcal{D}[\sigma_-]\rho + \gamma_\phi\,\mathcal{D}[\sigma_z]\rho,
$$

kde $\mathcal{D}[L]\rho = L\rho L^\dagger - \tfrac12\{L^\dagger L,\rho\}$.

### Co dělají ovladače
- **Maximum photon number** určuje ořez báze dutinového módu.
- **Coupling** a **Detuning** mají stejný význam jako v uzavřených Jaynesových–Cummingsových panelech.
- **Cavity loss** řídí únik fotonů z dutiny.
- **Atomic decay** převádí populaci z $|e\rangle$ do $|g\rangle$.
- **Pure dephasing** tlumí koherenci, aniž by přímo měnil atomové populace.
- **Maximum time** a **Animation frames** určují časové okno a časové vzorkování.

### Co ukazuje každý graf
- **Levý line plot:** populace excitovaného a základního stavu atomu.
- **Pravý line plot:** střední počet fotonů v dutině a celková čistota $\mathrm{Tr}(\rho^2)$.
- **Animated multi-panel figure:** stejné časové průběhy populací a počtu fotonů plus okamžité rozdělení počtu fotonů v poli.
- **Dodatečná animace čistoty:** čistota jako funkce času s pohybujícím se markerem.
- **Bloch sphere panel:** redukovaný stav atomu jako trajektorie uvnitř Blochovy koule.

### Jak tomu rozumět
V uzavřeném modelu je vývoj unitární a informace zůstává uvnitř systému atom–pole. Zde prostředí odvádí fotony i koherenci, takže oscilace se tlumí, čistota klesá a trajektorie Blochova vektoru se stahuje dovnitř koule. Cavity loss primárně odebírá excitaci z pole, atomic decay vybíjí stav $|e\rangle$ a dephasing hlavně ničí mimodiagonální koherenci. Porovnání tohoto panelu s uzavřenými vacuum-Rabi a collapse–revival panely nejlépe ukáže, co se ztratí, když prostředí už nelze zanedbat [8–10].

Reference: [5], [8]–[10].
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
        "cavity_loss": "Ztráty dutiny κ / ωc",
        "atomic_decay": "Atomový rozpad γ / ωc",
        "dephasing": "Čistý dephasing γφ / ωc",
        "purity": "Čistota Tr(ρ²)",
        "inst_field_dist": "Okamžité rozdělení počtu fotonů v poli",
        "selected_detuning": "Vybraný řez detuningem",
        "detuning_range": "Maximální |Δ| / ωc pro scan",
        "vacuum_note": "Počáteční stav je fixován na $|e,0\\rangle$.",
        "realizations_note": "Všechny výpočty používají bezrozměrné jednotky s $\\hbar = 1$ a $\\omega_c = 1$. Báze počtu fotonů je oříznuta na zvolený maximální počet fotonů.",
        "footer": "Rozsah modelu: jednómódová cavity-QED dynamika v Jaynesově–Cummingsově tvaru, a to jak pro uzavřený unitární vývoj, tak pro minimální otevřený Lindbladův model, implementovaná přes vektorizovanou lineární algebru NumPy/SciPy. Rozsah platnosti a reference jsou v teoretické části.",
    },
}

DEFAULTS = {
    "light": {"state_type": "coherent", "nmax": 20, "fock_n": 3, "alpha": 2.0, "nbar": 4.0},
    "jc": {"atom": "e", "n0": 2, "nmax": 12, "g": 0.08, "delta": 0.0, "tmax": 120.0, "frames": 48},
    "vacuum": {"nmax": 8, "g": 0.10, "delta": 0.0, "tmax": 120.0, "frames": 48},
    "revival": {"nmax": 30, "alpha": 4.0, "g": 0.05, "delta": 0.0, "tmax": 420.0, "frames": 64},
    "open": {"nmax": 10, "g": 0.08, "delta": 0.0, "kappa": 0.015, "gamma": 0.008, "gamma_phi": 0.004, "tmax": 140.0, "frames": 56},
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




@st.cache_resource
def lindblad_eigendecomposition(nmax: int, g: float, delta: float, kappa: float, gamma: float, gamma_phi: float):
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
    dim = H.shape[0]
    I = np.eye(dim, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    collapse_ops = []
    if kappa > 0:
        collapse_ops.append(np.sqrt(kappa) * np.kron(a, ident_a))
    if gamma > 0:
        collapse_ops.append(np.sqrt(gamma) * np.kron(ident_f, sigma_m))
    if gamma_phi > 0:
        collapse_ops.append(np.sqrt(gamma_phi) * np.kron(ident_f, sigma_z))

    for c in collapse_ops:
        cdgc = c.conj().T @ c
        L += np.kron(c.conj(), c)
        L -= 0.5 * np.kron(I, cdgc)
        L -= 0.5 * np.kron(cdgc.T, I)

    evals, evecs = eig(L)
    inv_evecs = np.linalg.inv(evecs)
    return evals, evecs, inv_evecs


@st.cache_data
def evolve_density_matrices(nmax: int, rho0: np.ndarray, g: float, delta: float, kappa: float, gamma: float, gamma_phi: float, times: np.ndarray):
    evals, evecs, inv_evecs = lindblad_eigendecomposition(nmax, float(g), float(delta), float(kappa), float(gamma), float(gamma_phi))
    dim = rho0.shape[0]
    rho0_vec = rho0.reshape(dim * dim, order="F")
    coeffs = inv_evecs @ rho0_vec
    phases = np.exp(np.outer(times, evals))
    rho_vecs = (phases * coeffs[None, :]) @ evecs.T
    rhos = rho_vecs.reshape(len(times), dim, dim, order="F")
    rhos = 0.5 * (rhos + np.conj(np.swapaxes(rhos, 1, 2)))
    traces = np.trace(rhos, axis1=1, axis2=2).real
    rhos = rhos / traces[:, None, None]
    return rhos


def observables_from_rhos(rhos: np.ndarray, nmax: int):
    diag = np.real(np.diagonal(rhos, axis1=1, axis2=2))
    diag = diag.reshape(len(rhos), nmax + 1, 2)
    pg = diag[:, :, 0].sum(axis=1)
    pe = diag[:, :, 1].sum(axis=1)
    n = np.arange(nmax + 1)[None, :]
    field_dist = diag.sum(axis=2)
    mean_n = (field_dist * n).sum(axis=1)
    purity = np.real(np.einsum("tij,tji->t", rhos, rhos))
    return pe, pg, mean_n, field_dist, purity


def bloch_vectors_from_rhos(rhos: np.ndarray, nmax: int):
    reshaped = rhos.reshape(len(rhos), nmax + 1, 2, nmax + 1, 2)
    rho_atom = np.zeros((len(rhos), 2, 2), dtype=complex)
    rho_atom[:, 0, 0] = np.sum(reshaped[:, :, 0, :, 0][:, np.arange(nmax + 1), np.arange(nmax + 1)], axis=1)
    rho_atom[:, 1, 1] = np.sum(reshaped[:, :, 1, :, 1][:, np.arange(nmax + 1), np.arange(nmax + 1)], axis=1)
    rho_atom[:, 0, 1] = np.sum(reshaped[:, :, 0, :, 1][:, np.arange(nmax + 1), np.arange(nmax + 1)], axis=1)
    rho_atom[:, 1, 0] = np.conj(rho_atom[:, 0, 1])
    bx = 2.0 * np.real(rho_atom[:, 0, 1])
    by = 2.0 * np.imag(rho_atom[:, 0, 1])
    bz = np.real(rho_atom[:, 1, 1] - rho_atom[:, 0, 0])
    return bx, by, bz


def observables_from_states(states: np.ndarray):
    probs = np.abs(states) ** 2
    pe = probs[:, :, 1].sum(axis=1)
    pg = probs[:, :, 0].sum(axis=1)
    n = np.arange(states.shape[1])[None, :]
    mean_n = (probs.sum(axis=2) * n).sum(axis=1)
    field_dist = probs.sum(axis=2)
    return pe, pg, mean_n, field_dist


def bloch_vectors(states: np.ndarray):
    rho_ge = np.sum(states[:, :, 0] * np.conj(states[:, :, 1]), axis=1)
    bx = 2.0 * np.real(rho_ge)
    by = 2.0 * np.imag(rho_ge)
    bz = np.sum(np.abs(states[:, :, 1]) ** 2 - np.abs(states[:, :, 0]) ** 2, axis=1)
    return bx, by, bz



def bloch_copy():
    if st.session_state.app_lang == "cs":
        return {
            "title": "Blochova koule atomového stavu",
            "body": r"""
Tato vizualizace ukazuje **redukovaný stav dvouhladinového atomu** ve formě bodu na Blochově kouli.

### Co zde vidíš
- severní pól $z=+1$ odpovídá stavu $|e\rangle$,
- jižní pól $z=-1$ odpovídá stavu $|g\rangle$,
- body na povrchu představují **čisté stavy**,
- body uvnitř koule představují **smíšené redukované stavy**.

### Jak tomu rozumět
V Jaynesově–Cummingsově dynamice se atom typicky **provazuje** s polem. Proto se redukovaný atomový stav často pohybuje **dovnitř koule**, ne jen po jejím povrchu. Když je atomový stav téměř čistý, trajektorie se blíží povrchu; když je atom silněji provázán s polem, redukovaný stav se stává více smíšeným a bod se posouvá dovnitř [3,4].

Matematicky jde o stejnou geometrii jako u **Poincarého koule** pro polarizaci, ale pro dvouúrovňový atom je standardním názvem **Blochova koule**.

Reference: [3], [4].
""",
        }
    return {
        "title": "Bloch sphere of the atomic state",
        "body": r"""
This visualization shows the **reduced state of the two-level atom** as a point inside the Bloch sphere.

### What is shown
- the north pole $z=+1$ corresponds to $|e\rangle$,
- the south pole $z=-1$ corresponds to $|g\rangle$,
- points on the surface represent **pure states**,
- points inside the sphere represent **mixed reduced states**.

### How to interpret it
In Jaynes–Cummings dynamics the atom generally becomes **entangled** with the field. That is why the reduced atomic state often moves **inside the sphere**, not only on its surface. When the atomic state is close to pure, the trajectory approaches the surface; when atom–field entanglement is stronger, the reduced atomic state becomes more mixed and the point moves inward [3,4].

Mathematically this is the same geometry as the **Poincaré sphere** for polarization, but for a two-level atom the standard name is the **Bloch sphere**.

References: [3], [4].
""",
    }


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




def animated_bloch_sphere(times, bx, by, bz):
    u = np.linspace(0.0, 2.0 * np.pi, 50)
    v = np.linspace(0.0, np.pi, 26)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    th = np.linspace(0.0, 2.0 * np.pi, 300)
    circle_xy = (np.cos(th), np.sin(th), np.zeros_like(th))
    circle_xz = (np.cos(th), np.zeros_like(th), np.sin(th))
    circle_yz = (np.zeros_like(th), np.cos(th), np.sin(th))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.14,
            showscale=False,
            colorscale=[[0.0, "#9ecae1"], [1.0, "#9ecae1"]],
            hoverinfo="skip",
            name=""
        )
    )
    for cx, cy, cz in (circle_xy, circle_xz, circle_yz):
        fig.add_trace(
            go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.55)", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            line=dict(color="rgba(80,80,80,0.28)", width=4),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=bx[:1], y=by[:1], z=bz[:1],
            mode="lines",
            line=dict(color="#1f77b4", width=6),
            name="trajectory",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[bx[0]], y=[by[0]], z=[bz[0]],
            mode="markers",
            marker=dict(size=6, color="#d62728"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1, 1],
            mode="markers+text",
            marker=dict(size=3, color=["#444444", "#444444"]),
            text=["|g⟩", "|e⟩"],
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    frames = []
    for k in range(len(times)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=bx[: k + 1], y=by[: k + 1], z=bz[: k + 1]),
                    go.Scatter3d(x=[bx[k]], y=[by[k]], z=[bz[k]]),
                ],
                traces=[5, 6],
                name=str(k),
            )
        )
    fig.frames = frames

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1], title="x", showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[-1.1, 1.1], title="y", showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(range=[-1.1, 1.1], title="z", showbackground=False, showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.15)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02,
                y=1.03,
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

    tabs = st.tabs([tr("tab_light"), tr("tab_jc"), tr("tab_vacuum"), tr("tab_revival"), tr("tab_open"), tr("tab_detuning")])

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

        bloch_text = bloch_copy()
        with st.expander(bloch_text["title"], expanded=False):
            st.markdown(bloch_text["body"])
        bx, by, bz = bloch_vectors(states)
        st.plotly_chart(animated_bloch_sphere(times, bx, by, bz), use_container_width=True)

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

        bloch_text = bloch_copy()
        with st.expander(bloch_text["title"], expanded=False):
            st.markdown(bloch_text["body"])
        bx, by, bz = bloch_vectors(states)
        st.plotly_chart(animated_bloch_sphere(times, bx, by, bz), use_container_width=True)

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

        bloch_text = bloch_copy()
        with st.expander(bloch_text["title"], expanded=False):
            st.markdown(bloch_text["body"])
        bx, by, bz = bloch_vectors(states)
        st.plotly_chart(animated_bloch_sphere(times, bx, by, bz), use_container_width=True)

    with tabs[4]:
        with st.expander(tr("help_open"), expanded=False):
            st.markdown(tr("help_open_body"))
        cols = st.columns([1.0, 1.0, 0.7])
        with cols[0]:
            st.slider(tr("nmax"), 4, 16, key="open_nmax")
            st.slider(tr("coupling"), 0.01, 0.25, step=0.01, key="open_g")
            st.slider(tr("detuning"), -0.40, 0.40, step=0.01, key="open_delta")
        with cols[1]:
            st.slider(tr("cavity_loss"), 0.0, 0.10, step=0.002, key="open_kappa")
            st.slider(tr("atomic_decay"), 0.0, 0.10, step=0.002, key="open_gamma")
            st.slider(tr("dephasing"), 0.0, 0.10, step=0.002, key="open_gamma_phi")
            st.slider(tr("tmax"), 20.0, 260.0, step=5.0, key="open_tmax")
            st.slider(tr("frames"), 24, 140, key="open_frames")
        with cols[2]:
            st.button(tr("reset"), key="reset_open", on_click=reset_section, args=("open",))

        times = np.linspace(0.0, st.session_state.open_tmax, int(st.session_state.open_frames))
        psi0 = basis_state(st.session_state.open_nmax, "e", 1 if st.session_state.open_nmax >= 1 else 0)
        rho0 = np.outer(psi0, np.conj(psi0))
        rhos = evolve_density_matrices(
            st.session_state.open_nmax,
            rho0,
            st.session_state.open_g,
            st.session_state.open_delta,
            st.session_state.open_kappa,
            st.session_state.open_gamma,
            st.session_state.open_gamma_phi,
            times,
        )
        pe, pg, mean_n, field_dist, purity = observables_from_rhos(rhos, st.session_state.open_nmax)

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
            fig2.add_trace(go.Scatter(x=times, y=purity, mode="lines", name=tr("purity"), yaxis="y2"))
            fig2.update_layout(height=380, xaxis_title=tr("time"), yaxis=dict(title=tr("mean_photons")), yaxis2=dict(title=tr("purity"), overlaying="y", side="right", range=[0, 1.02]), margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(animated_jc_figure(times, pe, pg, mean_n, field_dist), use_container_width=True)
        st.plotly_chart(animated_marker_figure(times, purity, tr("purity")), use_container_width=True)

        bloch_text = bloch_copy()
        with st.expander(bloch_text["title"], expanded=False):
            st.markdown(bloch_text["body"])
        bx, by, bz = bloch_vectors_from_rhos(rhos, st.session_state.open_nmax)
        st.plotly_chart(animated_bloch_sphere(times, bx, by, bz), use_container_width=True)

    with tabs[5]:
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
