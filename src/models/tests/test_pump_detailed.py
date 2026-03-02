import pytest
import os
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    units as pyunits,
    value,
    Reals,
)
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.constants import Constants
from pyomo.util.check_units import assert_units_consistent
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.core.solvers import get_solver
from models.pump_detailed import Pump, Efficiency, PumpCurveDataType


solver = get_solver()

# Build function with design flow and head as inputs
def build_pump_w_flow_head():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.SurrogateCoefficent,
        head_surrogate_coeffs={0: 114.22, 1: -410.6, 2: 2729.2, 3: -8089.1},
        efficiency_surrogate_coeffs={0: 0.389, 1: -0.535, 2: 41.373, 3: -138.82},
    )

    # Input flow and head
    feed_flow_vol = 0.126 * pyunits.m**3 / pyunits.s
    pump_head = 60.96 * pyunits.m
    density = 1000 * pyunits.kg / pyunits.m**3

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_TDS = 0.035
    feed_pressure_in = 101325 * pyunits.Pa
    feed_pressure_out = (
        feed_pressure_in + pump_head * density * Constants.acceleration_gravity
    )
    feed_temperature = 273.15 + 25

    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.outlet.pressure[0].fix(feed_pressure_out)

    m.fs.unit.system_curve_geometric_head.fix(4.57)
    m.fs.unit.ref_speed_fraction.fix(1.0)

    assert_units_consistent(m)

    return m


# Build function with design flow and speed as inputs
def build_pump_w_flow_speed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()
    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.SurrogateCoefficent,
        head_surrogate_coeffs={0: 114.22, 1: -410.6, 2: 2729.2, 3: -8089.1},
        efficiency_surrogate_coeffs={0: 0.389, 1: -0.535, 2: 41.373, 3: -138.82},
    )
    # Input flow and speed
    feed_flow_vol = 0.126 * pyunits.m**3 / pyunits.s
    design_speed_fraction = 0.829
    density = 1000 * pyunits.kg / pyunits.m**3

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_TDS = 0.035

    feed_pressure_in = 101325 * pyunits.Pa
    feed_temperature = 273.15 + 25

    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)

    m.fs.unit.design_speed_fraction.fix(design_speed_fraction)
    m.fs.unit.system_curve_geometric_head.fix(4.57)
    m.fs.unit.ref_speed_fraction.fix(1.0)

    calculate_scaling_factors(m)
    assert_units_consistent(m)

    return m


@pytest.mark.component
def test_fixed_eff_pump():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Fixed,
    )

    # Input flow and speed
    feed_flow_vol = 0.126 * pyunits.m**3 / pyunits.s
    density = 1000 * pyunits.kg / pyunits.m**3

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_TDS = 0.035

    feed_pressure_in = 101325 * pyunits.Pa
    feed_temperature = 273.15 + 25

    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)

    m.fs.unit.efficiency_pump.fix(0.85)
    m.fs.unit.deltaP.fix(500000)

    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert pytest.approx(m.fs.unit.work_mechanical[0].value, rel=1e-3) == 72411


# Three tests for different combinations of inputs for variable efficiency
@pytest.mark.component
def test_pump_w_flow_head():
    m = build_pump_w_flow_head()

    assert hasattr(m.fs.unit, "inlet")
    assert hasattr(m.fs.unit, "outlet")
    assert hasattr(m.fs.unit, "deltaP")  # this is just a reference
    assert hasattr(m.fs.unit.control_volume, "deltaP")

    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)
    assert_optimal_termination(results)


@pytest.mark.component
def test_pump_w_flow_speed():
    m = build_pump_w_flow_speed()

    assert hasattr(m.fs.unit, "inlet")
    assert hasattr(m.fs.unit, "outlet")
    assert hasattr(m.fs.unit, "deltaP")  # this is just a reference
    assert hasattr(m.fs.unit.control_volume, "deltaP")

    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)
    assert_optimal_termination(results)


@pytest.mark.component
def test_pump_w_head_speed():
    m = build_pump_w_flow_head()

    assert hasattr(m.fs.unit, "inlet")
    assert hasattr(m.fs.unit, "outlet")
    assert hasattr(m.fs.unit, "deltaP")  # this is just a reference
    assert hasattr(m.fs.unit.control_volume, "deltaP")

    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    m.fs.unit.design_speed_fraction.fix(0.8184)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.unit.control_volume.properties_in[0].mass_frac_phase_comp["Liq", "TDS"].fix()

    results = solver.solve(m)
    assert_optimal_termination(results)


# Test for passing a dataset for the pump curves
@pytest.mark.component
def test_data_points():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()
    pump_curves_filepath = os.path.join(
        os.path.dirname(__file__), "test_pump_curves_data.csv"
    )

    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.DataSet,
        # flow in m3/s and head in m
        pump_curves=pump_curves_filepath,
    )

    # Input flow and head
    feed_flow_vol = 0.1435 * pyunits.m**3 / pyunits.s
    pump_head = 87 * pyunits.m
    density = 1000 * pyunits.kg / pyunits.m**3

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_TDS = 0.035
    feed_pressure_in = 101325 * pyunits.Pa
    feed_pressure_out = (
        feed_pressure_in + pump_head * density * Constants.acceleration_gravity
    )
    feed_temperature = 273.15 + 25

    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.outlet.pressure[0].fix(feed_pressure_out)

    m.fs.unit.system_curve_geometric_head.fix(0)
    m.fs.unit.ref_speed_fraction.fix(1.0)

    assert hasattr(m.fs.unit, "inlet")
    assert hasattr(m.fs.unit, "surrogate_index")

    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)
    assert_optimal_termination(results)


@pytest.mark.component
def test_low_speed():
    # Doubles as low speed (50%) test
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
    feed_flow_vol = 0.12 * pyunits.m**3 / pyunits.s
    pump_head = 21.3 * pyunits.m
    density = 1000 * pyunits.kg / pyunits.m**3

    # Calculated feed conditions
    feed_flow_mass = feed_flow_vol * density
    feed_mass_frac_TDS = 0.035

    feed_pressure_in = 101325 * pyunits.Pa
    feed_pressure_out = feed_pressure_in + pump_head * density * Constants.acceleration_gravity
    feed_temperature = 273.15 + 25

    feed_mass_frac_H2O = 1 - feed_mass_frac_TDS
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        feed_flow_mass * feed_mass_frac_TDS
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure_in)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.outlet.pressure[0].fix(feed_pressure_out)

    m.fs.unit.system_curve_geometric_head.fix(0)
    m.fs.unit.ref_speed_fraction.fix(1.0)

    calculate_scaling_factors(m)
    m.fs.unit.initialize()
    assert degrees_of_freedom(m) == 0

    # Now apply the pump speed
    test_pump_speed = 0.5
    m.fs.unit.design_speed_fraction.fix(test_pump_speed)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.unit.control_volume.properties_in[0].mass_frac_phase_comp["Liq", "TDS"].fix()
    calculate_scaling_factors(m)

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert m.fs.unit.efficiency_pump[0].value == pytest.approx(0.64, abs=0.02)


@pytest.mark.component
def test_negative_inlet_pressure():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    m.fs.unit = Pump(
        property_package=m.fs.properties,
        variable_efficiency=Efficiency.Flow,
        pump_curve_data_type=PumpCurveDataType.DataSet,
        pump_curves=os.path.join(
            os.path.dirname(__file__),
            "test_pump_curves_data_uf.csv",
        ),
    )

    # Input flow and head for initial solve
    feed_flow_vol = (
        0.21 * pyunits.m**3 / pyunits.s
    )  # Actual flow vol will be calculated in second solve
    density = 1000 * pyunits.kg / pyunits.m**3
    feed_pressure_in = -12 * pyunits.psi  # 101325 * pyunits.Pa

    m.fs.unit.control_volume.properties_in[0].pressure.setlb(None)
    m.fs.unit.control_volume.properties_in[0].pressure.domain = Reals

    feed_pressure_out = 50 * pyunits.psi
    feed_mass_frac_TDS = 0.0005
    feed_temperature = 298.15
    geometric_head = pyunits.convert(
        12 * pyunits.psi / (density * Constants.acceleration_gravity),
        to_units=pyunits.m,
    )

    # Fix pump characteristics
    m.fs.unit.system_curve_geometric_head.fix(geometric_head)
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

    solver = get_solver()
    test_pump_speed = 0.91
    m.fs.unit.design_speed_fraction.fix(test_pump_speed)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.unit.control_volume.properties_in[0].mass_frac_phase_comp["Liq", "TDS"].fix()
    calculate_scaling_factors(m)
    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(
        pyunits.convert(m.fs.unit.work_mechanical[0], to_units=pyunits.kW)
    ) == pytest.approx(166.54, rel=1e-3)


# Test an invalid surrogate coefficient case
@pytest.mark.component
def test_invalid_surrogate_coefficients():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    with pytest.raises(
        ValueError,
        match=r"Surrogate coefficient keys must be exactly \{0, 1, 2, 3\}\.",
    ):
        m.fs.unit = Pump(
            property_package=m.fs.properties,
            variable_efficiency=Efficiency.Flow,
            pump_curve_data_type=PumpCurveDataType.SurrogateCoefficent,
            head_surrogate_coeffs={0: 114.22, 1: -410.6, 2: 2729.2},
            efficiency_surrogate_coeffs={0: 0.389, 1: -0.535},
        )

# Test an invalid filepath name
@pytest.mark.component
def test_invalid_filepath():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    with pytest.raises(
        ValueError,
        match=r"Failed to read CSV file '.*': .*",
    ):
        m.fs.unit = Pump(
            property_package=m.fs.properties,
            variable_efficiency=Efficiency.Flow,
            pump_curve_data_type=PumpCurveDataType.DataSet,
            pump_curves=os.path.join(os.path.dirname(__file__), "DNE.csv"),
        )


# Test missing filepath for dataset mode
@pytest.mark.component
def test_missing_filepath_for_dataset_mode():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    with pytest.raises(
        ConfigurationError,
        match=r"pump_curves must be provided as a CSV filepath when pump_curve_data_type is DataSet\.",
    ):
        m.fs.unit = Pump(
            property_package=m.fs.properties,
            variable_efficiency=Efficiency.Flow,
            pump_curve_data_type=PumpCurveDataType.DataSet,
            pump_curves=None,
        )


@pytest.mark.component
def test_invalid_efficiency_type():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    with pytest.raises(
        ValueError,
        match="'InvalidType' is not a valid Efficiency",
    ):
        m.fs.unit = Pump(
            property_package=m.fs.properties,
            variable_efficiency="InvalidType",
            pump_curve_data_type=PumpCurveDataType.SurrogateCoefficent,
        )


@pytest.mark.component
def test_missing_surrogates():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()

    with pytest.raises(
        ConfigurationError,
        match="surrogate_coeffs must be provided for the pump head curve and efficiency curve when pump_curve_data_type is set to SurrogateCoefficent.",
    ):
        m.fs.unit = Pump(
            property_package=m.fs.properties,
            variable_efficiency=Efficiency.Flow,
            pump_curve_data_type=PumpCurveDataType.SurrogateCoefficent,
        )

