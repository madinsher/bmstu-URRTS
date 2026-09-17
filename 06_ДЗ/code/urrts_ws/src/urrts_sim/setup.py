from setuptools import setup

package_name = "urrts_sim"
setup(
    name=package_name,
    version="2026.1.0",
    packages=[package_name],
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"])],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="URRTS course",
    description="Kinematic warehouse simulator",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["sim2d = urrts_sim.sim2d_node:main"]},
)
