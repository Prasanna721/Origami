"""Material definitions with derived stiffness parameters."""
from dataclasses import dataclass


@dataclass
class Material:
    name: str
    thickness_mm: float
    youngs_modulus_gpa: float
    max_strain: float
    poisson_ratio: float = 0.3
    density_kg_m3: float = 1000.0

    @property
    def thickness_m(self) -> float:
        return self.thickness_mm / 1000.0

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_gpa * 1e9

    @property
    def k_axial(self) -> float:
        """Axial stiffness coefficient for bar constraints."""
        return self.youngs_modulus_pa * self.thickness_m

    @property
    def k_facet(self) -> float:
        """Facet bending stiffness for facet hinge constraints."""
        t = self.thickness_m
        E = self.youngs_modulus_pa
        nu = self.poisson_ratio
        return E * t ** 3 / (12.0 * (1.0 - nu ** 2))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "thickness_mm": self.thickness_mm,
            "youngs_modulus_gpa": self.youngs_modulus_gpa,
            "max_strain": self.max_strain,
            "poisson_ratio": self.poisson_ratio,
            "density_kg_m3": self.density_kg_m3,
        }


MATERIALS = {
    "paper": Material(
        name="paper",
        thickness_mm=0.1,
        youngs_modulus_gpa=3.0,
        max_strain=0.05,
        poisson_ratio=0.3,
        density_kg_m3=700.0,
    ),
    "mylar": Material(
        name="mylar",
        thickness_mm=0.05,
        youngs_modulus_gpa=4.0,
        max_strain=0.03,
        poisson_ratio=0.38,
        density_kg_m3=1390.0,
    ),
    "aluminum": Material(
        name="aluminum",
        thickness_mm=0.2,
        youngs_modulus_gpa=70.0,
        max_strain=0.01,
        poisson_ratio=0.33,
        density_kg_m3=2700.0,
    ),
    "nitinol": Material(
        name="nitinol",
        thickness_mm=0.15,
        youngs_modulus_gpa=75.0,
        max_strain=0.08,
        poisson_ratio=0.33,
        density_kg_m3=6450.0,
    ),
}
