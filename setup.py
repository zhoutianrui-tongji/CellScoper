from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import pathlib

here = pathlib.Path(__file__).parent.resolve()
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else "CellScoper: Spatial transcriptomics pipeline."

_BUNDLED_FILES = (
	(here / "config" / "params.yaml", "params.yaml"),
	(here / "src" / "2019_CELL_APEXSeq.tsv", "2019_CELL_APEXSeq.tsv"),
	(here / "src" / "RNALocateV3.txt", "RNALocateV3.txt"),
	(here / "src" / "extra_reference_markers.tsv", "extra_reference_markers.tsv"),
)


class build_py(_build_py):
	"""Copy runtime resources into the importable package in built wheels."""

	def run(self):
		super().run()
		target = pathlib.Path(self.build_lib) / "cellscoper" / "data"
		target.mkdir(parents=True, exist_ok=True)
		for source, name in _BUNDLED_FILES:
			self.copy_file(str(source), str(target / name))

	def get_outputs(self, include_bytecode=1):
		outputs = list(super().get_outputs(include_bytecode=include_bytecode))
		target = pathlib.Path(self.build_lib) / "cellscoper" / "data"
		outputs.extend(str(target / name) for _, name in _BUNDLED_FILES)
		return outputs

setup(
	name="CellScoper",
	version="1.0.0",
	description="Subcellular structure discovery and annotation for imaging-based spatial transcriptomics",
	long_description=readme,
	long_description_content_type="text/markdown",
	author="CellScoper Contributors",
	url="https://github.com/zhoutianrui-tongji/CellScoper",
	license="MIT",
	packages=find_packages(),
	cmdclass={"build_py": build_py},
	python_requires=">=3.9",
	install_requires=[
		"anndata>=0.10",
		"pandas>=1.5",
		"numpy>=1.24",
		"rich>=13",
		"click>=8",
		"pyarrow>=14",
		"shapely>=2.0",
		"scikit-learn>=1.3",  # sklearn.cluster.HDBSCAN (MetaDomain discovery)
		"scipy>=1.10",
		"joblib>=1.3",
		"matplotlib>=3.7",
		"PyYAML>=6.0",
		"threadpoolctl>=3.1",
		# LENS (masked-transcript model); a CUDA GPU makes it practical beyond a few thousand cells
		"torch>=2.1",
	],
	entry_points={
		"console_scripts": [
			"cellscoper=cellscoper.cli:main",
		]
	},
	classifiers=[
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Bio-Informatics",
	],
)
