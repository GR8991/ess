import streamlit as st

class BESSDesign:
    def __init__(self):
        self.losses = {}
        self.yearly_soh_percentages = []
        self.total_energy_mwh = 0
        self.usable_energy_mwh = 0
        self.yearly_usable_energies = []

    def get_basic_specs(self):
        self.mw = st.number_input("Enter MW", min_value=0.0)
        self.mwh = st.number_input("Enter MWh", min_value=0.0)
        self.p_rate = self.mw / self.mwh if self.mwh else 0
        self.hours = self.mwh / self.mw if self.mw else 0

    def get_cell_specs(self):
        self.cell_voltage = st.number_input("Enter cell voltage", min_value=0.0)
        self.cell_ah = st.number_input("Enter the cell Ah", min_value=0.0)

    def create_pack(self):
        e = st.number_input("Enter no. of parallel connection for pack", min_value=0.0)
        f = st.number_input("Enter no. of series connection for pack", min_value=0.0)
        self.pack_kwh = (e * self.cell_ah * f * self.cell_voltage) / 1000

    def create_rack(self):
        g = st.number_input("Enter no. of parallel connection for rack", min_value=0.0)
        h = st.number_input("Enter no. of series connection for rack", min_value=0.0)
        self.rack_kwh = (g * self.cell_ah * h * self.cell_voltage) / 1000

    def create_container(self):
        i = st.number_input("Enter no. of racks for container", min_value=0.0)
        self.container_mwh = (self.rack_kwh * i) / 1000

    def create_unit(self):
        k = st.number_input("Enter no. of containers connected to unit", min_value=0.0)
        self.unit_mwh = k * self.container_mwh

    def get_lifecycle_inputs(self):
        self.cycles = st.number_input("Enter no. of cycles", min_value=0.0)
        self.years = st.number_input("Enter no. of years", min_value=1, step=1, format="%d")
        self.dc_usable_ratio = self.get_percentage_input("Enter DC usable ratio")
        self.calendar_degradation = self.get_percentage_input("Enter calendar degradation")

        self.yearly_soh_percentages = []
        st.markdown("### Enter SOH (State of Health in %) for each year")
        for year in range(1, int(self.years) + 1):
            soh = self.get_percentage_input(f"Year {year} SOH")
            self.yearly_soh_percentages.append(soh)

    def get_percentage_input(self, prompt):
        return st.slider(prompt, min_value=0.0, max_value=100.0, value=100.0)

    def get_losses(self):
        voltage = st.selectbox("Select voltage level (kV)", options=[33, 66, 132, 220])

        st.markdown("### Enter the retained efficiency (%) for the following components")
        if voltage == 33:
            self.losses = {
                "DC cable": self.get_percentage_input("DC cable (e.g., 99.85 for 0.15% loss)"),
                "LV AC cable": self.get_percentage_input("LV AC cable"),
                "PCS efficiency": self.get_percentage_input("PCS efficiency"),
                "LV transformer": self.get_percentage_input("LV transformer"),
                "MV AC cable": self.get_percentage_input("MV AC cable"),
                "Measurement": self.get_percentage_input("Measurement"),
                "Availability": self.get_percentage_input("Availability")
            }
        else:
            self.losses = {
                "DC cable": self.get_percentage_input("DC cable"),
                "LV AC cable": self.get_percentage_input("LV AC cable"),
                "PCS efficiency": self.get_percentage_input("PCS efficiency"),
                "LV transformer": self.get_percentage_input("LV transformer"),
                "MV AC cable": self.get_percentage_input("MV AC cable"),
                "MV transformer": self.get_percentage_input("MV transformer"),
                "Transmission line": self.get_percentage_input("Transmission line"),
                "Measurement": self.get_percentage_input("Measurement"),
                "Availability": self.get_percentage_input("Availability")
            }

    def get_total_project_energy(self):
        num_units = st.number_input("Enter number of units deployed in project", min_value=1, step=1)
        self.total_energy_mwh = self.unit_mwh * num_units

    def calculate_usable_soh_excluded(self):
        if not self.yearly_soh_percentages:
            st.error("SOH data not available. Please enter lifecycle inputs first.")
            return

        efficiency_product = 1.0
        for val in self.losses.values():
            efficiency_product *= (val / 100)

        dc_usable = self.dc_usable_ratio / 100
        calendar_multiplier = self.calendar_degradation / 100

        usable_energy_base = (
            self.total_energy_mwh *
            dc_usable *
            calendar_multiplier *
            efficiency_product
        )

        self.yearly_usable_energies = []
        st.markdown("## Yearly Usable Energy After SOH Degradation")
        data = []
        for year, soh_percent in enumerate(self.yearly_soh_percentages, start=1):
            soh_multiplier = soh_percent / 100
            usable_energy = usable_energy_base * soh_multiplier
            self.yearly_usable_energies.append(usable_energy)
            data.append((year, soh_percent, usable_energy))

        self.usable_energy_mwh = self.yearly_usable_energies[0]
        st.dataframe(data, columns=["Year", "SOH (%)", "Usable Energy (MWh)"])

    def display_summary(self):
        st.markdown("## SYSTEM SUMMARY")
        st.write(f"MW: {self.mw}")
        st.write(f"MWh: {self.mwh}")
        st.write(f"Power Rate (MW/MWh): {self.p_rate:.2f}")
        st.write(f"Hours of Operation (MWh/MW): {self.hours:.2f}")
        st.write(f"Pack Energy (kWh): {self.pack_kwh:.2f}")
        st.write(f"Rack Energy (kWh): {self.rack_kwh:.2f}")
        st.write(f"Container Energy (MWh): {self.container_mwh:.2f}")
        st.write(f"Unit Energy (MWh): {self.unit_mwh:.2f}")
        st.write(f"Cycles: {self.cycles:.2f}")
        st.write(f"Years: {self.years}")
        st.write(f"DC Usable Ratio: {self.dc_usable_ratio}%")
        st.write(f"Calendar Degradation: {self.calendar_degradation}%")

        st.markdown("### Efficiency Factors (%)")
        for name, val in self.losses.items():
            st.write(f"{name}: {val}%")

        st.write(f"Total Energy Deployed (All Units): {self.total_energy_mwh:.2f} MWh")
        st.write(f"Usable Energy (SOH excluded) Year 1: {self.usable_energy_mwh:.2f} MWh")


def main():
    st.title("BESS Design Calculator")
    bess = BESSDesign()
    if st.button("Run BESS Calculation"):
        bess.get_basic_specs()
        bess.get_cell_specs()
        bess.create_pack()
        bess.create_rack()
        bess.create_container()
        bess.create_unit()
        bess.get_lifecycle_inputs()
        bess.get_losses()
        bess.get_total_project_energy()
        bess.calculate_usable_soh_excluded()
        bess.display_summary()


if __name__ == "__main__":
    main()
