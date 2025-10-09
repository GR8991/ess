import pandas as pd
import numpy as np

# === USER-PROVIDED PARAMETERS ===
# 1. CSV file path
csv_file = '2025-10-02T07-29_export.csv'

# 2. Transformer parameters
k_loss    = 0.0062     # MW per MVA
Q_trafo   = 4.737      # MVAR

# 3. PCS voltage base
V_base    = 690        # Volts

# === END USER INPUT ===

# Read input CSV
df = pd.read_csv(csv_file)

# Prepare results storage
results = []

for scenario in ['Dis-Over','Cha-Over','Dis-Under','Cha-Under']:
    row = {'Scenario': scenario}
    idx = df.columns.get_loc(scenario)
    
    # MV grid inputs
    P_mv = float(df.loc[df['Parameter']=='P required on MV grid', scenario])
    Q_mv = float(df.loc[df['Parameter']=='Q required on MV grid', scenario])
    
    # Step 1: Apparent power at MV level
    S_mv = np.hypot(P_mv, Q_mv)
    
    # Step 2: Transformer losses & reactive demand
    P_loss = k_loss * S_mv
    P_pcs  = P_mv + P_loss
    Q_pcs  = Q_mv + Q_trafo
    
    # Step 3: PCS apparent power, pf, sinφ
    S_pcs    = np.hypot(P_pcs, Q_pcs)
    pf_pcs   = P_pcs / S_pcs
    sin_phi  = Q_pcs / S_pcs
    
    # Step 4: Determine operating mode
    if abs(Q_pcs) > 20:
        U_term_pu, k_ratio = 0.94, 0.22
    else:
        U_term_pu, k_ratio = 0.885, 0.45
    
    # Step 5: Power angle φ, voltage angle δ
    phi_rad   = np.arcsin(abs(sin_phi)) * np.sign(sin_phi)
    delta_rad = phi_rad * k_ratio
    
    # Step 6: DQ voltages
    Ud_pu = U_term_pu * np.cos(delta_rad)
    Uq_pu = U_term_pu * np.sin(delta_rad)
    
    # Step 7: Convert to volts and calculate U_terminal
    Ud_V   = Ud_pu * V_base
    Uq_V   = Uq_pu * V_base
    Uterm_pu = np.hypot(Ud_pu, Uq_pu)
    Uterm_V  = U_term_pu * V_base
    
    # Collect results
    row.update({
        'P_mv (MW)':    P_mv,
        'Q_mv (MVAR)':  Q_mv,
        'S_mv (MVA)':   S_mv,
        'P_loss (MW)':  P_loss,
        'P_pcs (MW)':   P_pcs,
        'Q_pcs (MVAR)': Q_pcs,
        'S_pcs (MVA)':  S_pcs,
        'pf_pcs':       pf_pcs,
        'sin_phi':      sin_phi,
        'U_term_pu':    U_term_pu,
        'Ud_pu':        Ud_pu,
        'Uq_pu':        Uq_pu,
        'Ud_V':         Ud_V,
        'Uq_V':         Uq_V,
        'Uterm_pu':     Uterm_pu
    })
    results.append(row)

# Create a results DataFrame
df_results = pd.DataFrame(results)

# Reorder columns for clarity
cols = [
    'Scenario',
    'P_mv (MW)','Q_mv (MVAR)','S_mv (MVA)',
    'P_loss (MW)',
    'P_pcs (MW)','Q_pcs (MVAR)','S_pcs (MVA)',
    'pf_pcs','sin_phi',
    'U_term_pu','Uterm_pu',
    'Ud_pu','Uq_pu',
    'Ud_V','Uq_V'
]
df_results = df_results[cols]

# Output the full table
print(df_results.to_string(index=False))
