from setuptools import setup

package_name = "urrts_fleet"
setup(
    name=package_name,
    version="2026.1.0",
    packages=[package_name],
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"])],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="URRTS course",
    description="Fleet logic: orders, CBBA, executors",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "order_source = urrts_fleet.order_source_node:main",
        "cbba = urrts_fleet.cbba_node:main",
        "executor = urrts_fleet.executor_node:main",
        "traffic = urrts_fleet.traffic_node:main",
    ]},
)
