from glob import glob

from setuptools import setup

package_name = "urrts_webots"
setup(
    name=package_name,
    version="2026.1.0",
    packages=[package_name],
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"]),
                ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
                ("share/" + package_name + "/config", glob("config/*.yaml"))],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="URRTS course",
    description="Webots warehouse and Nav2 bridge",
    license="MIT",
    entry_points={"console_scripts": [
        "bridge = urrts_webots.bridge_node:main",
        "world = urrts_webots.world_node:main",
    ]},
)
