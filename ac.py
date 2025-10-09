import numpy as np
import pandas as pd

def pcs_calculator(P_mv, Q_mv, k_loss=0.0062, Q_trafo=4.737, V_base=690):
    # Step 1: MV apparent power
    S_mv = np.hypot(P_mv, Q_mv)
    # Step 2: Transformer effects
    P_loss = k_loss * S_mv
    P_pcs = P_mv + P_loss
    Q_pcs = Q_mv + Q_trafo
    # Step 3: PCS power triangle
    S_pcs = np.hypot(P_pcs, Q_pcs)
    pf_pcs = P_pcs / S_pcs
    sin_phi = Q_pcs / S_pcs
    # Step 4: DQ voltage mode
    if abs(Q_pcs) > 20:
        U_term_pu, k_ratio = 0.94, 0.22
    else:
        U_term_pu, k_ratio = 0.885, 0.45
    # Step 5: Angles & DQ components
    phi = np.arcsin(abs(sin_phi)) * np.sign(sin_phi)
    delta = phi * k_ratio
    Ud_pu = U_term_pu * np.cos(delta)
    Uq_pu = U_term_pu * np.sin(delta)
    # Step 6: Convert to volts
    return {
        'P_pcs': P_pcs, 'Q_pcs': Q_pcs, 'S_pcs': S_pcs,
        'pf_pcs': pf_pcs, 'sin_phi': sin_phi,
        'U_term_pu': U_term_pu, 'Ud_pu': Ud_pu, 'Uq_pu': Uq_pu,
        'Ud_V': Ud_pu * V_base, 'Uq_V': Uq_pu * V_base
    }

# Prompt user for the four scenarios
scenarios = ['Dis-Over','Cha-Over','Dis-Under','Cha-Under']
inputs = {}
for s in scenarios:
    P = float(input(f"Enter P_mv for {s} (MW): "))
    Q = float(input(f"Enter Q_mv for {s} (MVAR): "))
    inputs[s] = pcs_calculator(P, Q)

# Build output table matching your CSV format columns
df = pd.DataFrame({
    'Parameter': [
        'P_pcs (MW)','Q_pcs (MVAR)','S_pcs (MVA)',
        'pf_pcs','sin_phi','U_term_pu','Ud_pu','Uq_pu'
    ]
})
for s in scenarios:
    vals = inputs[s]
    df[s] = [
        vals['P_pcs'], vals['Q_pcs'], vals['S_pcs'],
        vals['pf_pcs'], vals['sin_phi'], vals['U_term_pu'],
        vals['Ud_pu'], vals['Uq_pu']
    ]

print("\nPCS Calculation Results:")
print(df.to_string(index=False))
