# file is meant to be moved into the case study days test files in the Testing PR
# Comparison of the uf pumps on the two days.

import pytest
import os
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    units as pyunits,
    value,
)
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors
from pyomo.util.check_units import assert_units_consistent
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.core.solvers import get_solver
from models.pump_detailed import Pump, Efficiency, PumpCurveDataType

solver = get_solver()


@pytest.mark.unit
def test_uf_pump_8_19_full_speed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.DataSet,
        pump_curves=os.path.join(
            os.path.dirname(__file__), "test_pump_curves_data_uf.csv"
        ),
    )

    # Input flow and head for initial solve
    feed_flow_vol = 0.21 * pyunits.m**3 / pyunits.s # Actual flow vol will be calculated in second solve
    density = 1000 * pyunits.kg / pyunits.m**3
    feed_pressure_in = 101325 * pyunits.Pa
    m.fs.unit.control_volume.properties_in[0].pressure.setlb(None)
    m.fs.unit.control_volume.properties_in[0].pressure.setub(None)
    feed_pressure_out = 50 * pyunits.psi
    # Alternatively, calculate the outlet pressure from the head and inlet pressure 
    # pump_head = (84/3.28) * pyunits.m
    # feed_pressure_out = feed_pressure_in + pump_head * density * (9.81 * pyunits.m / pyunits.s**2)
    feed_mass_frac_TDS = 0.0005
    feed_temperature = 298.15

    # Fix pump characteristics
    m.fs.unit.system_curve_geometric_head.fix(0 * pyunits.ft)
    m.fs.unit.ref_speed_fraction.fix(1.0)
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.outlet.pressure[0].fix(feed_pressure_out)

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )

    calculate_scaling_factors(m)
    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0
    
    # Now apply the pump speed
    test_pump_speed = 0.91
    m.fs.unit.design_speed_fraction.fix(test_pump_speed)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.unit.control_volume.properties_in[0].mass_frac_phase_comp["Liq", "TDS"].fix()
    calculate_scaling_factors(m)
    results = solver.solve(m)
    assert_optimal_termination(results)
    expected_power = 175 #kW
    
    assert value(pyunits.convert(m.fs.unit.work_mechanical[0], to_units=pyunits.kW)) == pytest.approx(expected_power, rel=0.02)


@pytest.mark.unit
def test_uf_pump_8_19_low_speed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()
    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.DataSet,
        pump_curves=os.path.join(
            os.path.dirname(__file__), "test_pump_curves_data_uf.csv"
        ),
    )

    # Input flow and head for initial solve
    feed_flow_vol = 0.21 * pyunits.m**3 / pyunits.s # Actual flow vol will be calculated in second solve
    density = 1000 * pyunits.kg / pyunits.m**3
    feed_pressure_in = 101325 * pyunits.Pa
    # feed_pressure_out = 50 * pyunits.psi 
    pump_head = (100/3.28) * pyunits.m
    feed_pressure_out = feed_pressure_in + pump_head * density * (9.81 * pyunits.m / pyunits.s**2)
    feed_mass_frac_TDS = 0.0005
    feed_temperature = 298.15

    # Fix pump characteristics
    m.fs.unit.system_curve_geometric_head.fix(0 * pyunits.ft)
    m.fs.unit.ref_speed_fraction.fix(1.0)
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.outlet.pressure[0].fix(feed_pressure_out)

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )

    calculate_scaling_factors(m)
    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0
    # assert m.fs.unit.design_speed_fraction.value == pytest.approx(.696, rel=0.05)

    # Now apply the pump speed
    test_pump_speed = 0.75
    m.fs.unit.design_speed_fraction.fix(test_pump_speed)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.unit.control_volume.properties_in[0].mass_frac_phase_comp["Liq", "TDS"].fix()
    # m.fs.unit.vfd_efficiency.set_value(0.97)
    calculate_scaling_factors(m)

    results = solver.solve(m)
    assert_optimal_termination(results)
    expected_power = 80 # kW
    assert value(pyunits.convert(m.fs.unit.work_mechanical[0], to_units=pyunits.kW)) == pytest.approx(expected_power, rel=0.02)
    # assert value(m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"]) == pytest.approx(.2, rel=0.02)