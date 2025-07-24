<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org).

## [v2.1.0](https://github.com/neutrinoceros/gpgi/tree/v2.1.0) - 2025-07-24

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v2.0.0...v2.1.0)

- WHL: add wheels for Linux ARM64 [#385](https://github.com/neutrinoceros/gpgi/pull/385) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#381](https://github.com/neutrinoceros/gpgi/pull/381) ([pre-commit-ci](https://github.com/pre-commit-ci))
- TST: simplify bleeding-edge CI [#380](https://github.com/neutrinoceros/gpgi/pull/380) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: don’t skip tests on musllinux [#376](https://github.com/neutrinoceros/gpgi/pull/376) ([neutrinoceros](https://github.com/neutrinoceros))
- CLN: cleanup references to removed modules [#375](https://github.com/neutrinoceros/gpgi/pull/375) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: test wheels against Python 3.14 [#374](https://github.com/neutrinoceros/gpgi/pull/374) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: fix incorrect uv settings in cp314t tests [#368](https://github.com/neutrinoceros/gpgi/pull/368) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: upgrade pre-commit hooks [#367](https://github.com/neutrinoceros/gpgi/pull/367) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: start testing against CPython 3.14 [#366](https://github.com/neutrinoceros/gpgi/pull/366) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: complete thread-safety documentation [#365](https://github.com/neutrinoceros/gpgi/pull/365) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: upgrade cibuildwheel to 3.0.0 [#363](https://github.com/neutrinoceros/gpgi/pull/363) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: lockfile maintenance [#362](https://github.com/neutrinoceros/gpgi/pull/362) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: switch back to stable versions of Cython [#361](https://github.com/neutrinoceros/gpgi/pull/361) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: revert involuntary internal var renaming [#355](https://github.com/neutrinoceros/gpgi/pull/355) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: `Grid` and `ParticleSet` now report *all* invalid inputs instead of just the first one they find [#354](https://github.com/neutrinoceros/gpgi/pull/354) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#353](https://github.com/neutrinoceros/gpgi/pull/353) ([pre-commit-ci](https://github.com/pre-commit-ci))
- TYP: fix missing type arguments for generic types [#343](https://github.com/neutrinoceros/gpgi/pull/343) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: auto-fix TC006 violations [#342](https://github.com/neutrinoceros/gpgi/pull/342) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: report as many exceptions as possible from `gpgi.load` instead of just the first one encountered [#339](https://github.com/neutrinoceros/gpgi/pull/339) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: disable uv cache pruning in most CI [#332](https://github.com/neutrinoceros/gpgi/pull/332) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: enable building abi3 wheels (take 2, meson flavor) [#331](https://github.com/neutrinoceros/gpgi/pull/331) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: update badges in README [#330](https://github.com/neutrinoceros/gpgi/pull/330) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: trigger wheel builds on modifications to all build-related files [#329](https://github.com/neutrinoceros/gpgi/pull/329) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: report coverage for all files, including 100% covered ones [#327](https://github.com/neutrinoceros/gpgi/pull/327) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: adopt -ra pytest flags [#326](https://github.com/neutrinoceros/gpgi/pull/326) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: separate uv sync from uv run [#325](https://github.com/neutrinoceros/gpgi/pull/325) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: bump uv.lock [#324](https://github.com/neutrinoceros/gpgi/pull/324) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: disable uv caching for future deps tests [#323](https://github.com/neutrinoceros/gpgi/pull/323) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: add explicit mention of the GIL's state to pytest header [#322](https://github.com/neutrinoceros/gpgi/pull/322) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: drop QuansightLabs/setup-python action [#321](https://github.com/neutrinoceros/gpgi/pull/321) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: cleanup unused TypeVar [#319](https://github.com/neutrinoceros/gpgi/pull/319) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: bump uv.lock [#315](https://github.com/neutrinoceros/gpgi/pull/315) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: refine type hints to reflect relationships between arguments and return values in terms of array dtype [#314](https://github.com/neutrinoceros/gpgi/pull/314) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: rename private classes [#313](https://github.com/neutrinoceros/gpgi/pull/313) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: avoid importing enum as a namespace [#312](https://github.com/neutrinoceros/gpgi/pull/312) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: also typecheck against pyright [#311](https://github.com/neutrinoceros/gpgi/pull/311) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: fixup Dataset's docstring [#310](https://github.com/neutrinoceros/gpgi/pull/310) ([neutrinoceros](https://github.com/neutrinoceros))
- BUG: fix an error message that could incorrectly refer to the specific class, possibly not involved in the error itself [#309](https://github.com/neutrinoceros/gpgi/pull/309) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: goodbye inheritance, hello composition [#308](https://github.com/neutrinoceros/gpgi/pull/308) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: update uv configuration in bleeding-edge CI [#307](https://github.com/neutrinoceros/gpgi/pull/307) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: fix incorrect pre-commit hook tag [#306](https://github.com/neutrinoceros/gpgi/pull/306) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#305](https://github.com/neutrinoceros/gpgi/pull/305) ([pre-commit-ci](https://github.com/pre-commit-ci))
- MNT: update renovate schedule [#302](https://github.com/neutrinoceros/gpgi/pull/302) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: update renovate config and drop dependabot [#301](https://github.com/neutrinoceros/gpgi/pull/301) ([neutrinoceros](https://github.com/neutrinoceros))
- Update astral-sh/setup-uv action to v5 [#299](https://github.com/neutrinoceros/gpgi/pull/299) ([renovate](https://github.com/renovate))
- Update actions/upload-artifact action to v4.5.0 [#298](https://github.com/neutrinoceros/gpgi/pull/298) ([renovate](https://github.com/renovate))
- Update pypa/gh-action-pypi-publish action to v1.12.3 [#297](https://github.com/neutrinoceros/gpgi/pull/297) ([renovate](https://github.com/renovate))
- MNT: add uv-lock to pre-commit [#293](https://github.com/neutrinoceros/gpgi/pull/293) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: spread field maps validation logic into smaller functions [#291](https://github.com/neutrinoceros/gpgi/pull/291) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: fix type annotations for `_CoordinateValidatorMixin._get_safe_datatype` for compatibility with numpy 2.2 [#289](https://github.com/neutrinoceros/gpgi/pull/289) ([neutrinoceros](https://github.com/neutrinoceros))
- DEP: bump numpy to 2.2.0 [#288](https://github.com/neutrinoceros/gpgi/pull/288) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test against CPython 3.11.0 (instead of latest 3.11.x) [#287](https://github.com/neutrinoceros/gpgi/pull/287) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: rewrite testing workflows around `uv.lock` [#286](https://github.com/neutrinoceros/gpgi/pull/286) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: Update github actions [#284](https://github.com/neutrinoceros/gpgi/pull/284) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: cleanup unused CI step [#283](https://github.com/neutrinoceros/gpgi/pull/283) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: avoid pyplot interface in tests [#282](https://github.com/neutrinoceros/gpgi/pull/282) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: migrate requirement files to PEP 735 dependency groups [#279](https://github.com/neutrinoceros/gpgi/pull/279) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: prepare release 2.0.0 [#277](https://github.com/neutrinoceros/gpgi/pull/277) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: migrate to the mesonpy build backend [#269](https://github.com/neutrinoceros/gpgi/pull/269) ([neutrinoceros](https://github.com/neutrinoceros))

## [v2.0.0](https://github.com/neutrinoceros/gpgi/tree/v2.0.0) - 2024-10-09

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v1.0.0...v2.0.0)

### Added

- API: forbid overrides in `BoundaryRegistry.register` unless unsafe mutations are explicitly allowed. [#240](https://github.com/neutrinoceros/gpgi/pull/240) ([neutrinoceros](https://github.com/neutrinoceros))

### Fixed

- MNT: fixup coverage reporting [#264](https://github.com/neutrinoceros/gpgi/pull/264) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: document thread-safety of `BoundaryRegistry` [#263](https://github.com/neutrinoceros/gpgi/pull/263) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: add missing versionadded/versionchanged to docstrings [#261](https://github.com/neutrinoceros/gpgi/pull/261) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: Fix toc in documentation and a typo [#255](https://github.com/neutrinoceros/gpgi/pull/255) ([avirsaha](https://github.com/avirsaha))
- DOC: add changelog [#249](https://github.com/neutrinoceros/gpgi/pull/249) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: fixup internal logic in concurrency test [#242](https://github.com/neutrinoceros/gpgi/pull/242) ([neutrinoceros](https://github.com/neutrinoceros))
- BUG: fix thread safety for `BoundaryRegistry.register` [#241](https://github.com/neutrinoceros/gpgi/pull/241) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: auto fix typos [#226](https://github.com/neutrinoceros/gpgi/pull/226) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- TST: simplify CI [#278](https://github.com/neutrinoceros/gpgi/pull/278) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: prepare release 2.0.0 [#277](https://github.com/neutrinoceros/gpgi/pull/277) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: add CPython 3.13 to regular test matrix [#275](https://github.com/neutrinoceros/gpgi/pull/275) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#274](https://github.com/neutrinoceros/gpgi/pull/274) ([pre-commit-ci](https://github.com/pre-commit-ci))
- MNT: unpin uv [#272](https://github.com/neutrinoceros/gpgi/pull/272) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: pin uv to 0.4.9 [#270](https://github.com/neutrinoceros/gpgi/pull/270) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: simplify setup-uv usage following 2.1.1 release [#268](https://github.com/neutrinoceros/gpgi/pull/268) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: migrate to official astral-sh/setup-uv action (cd.yml) [#266](https://github.com/neutrinoceros/gpgi/pull/266) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: migrate to official astral-sh/setup-uv action [#265](https://github.com/neutrinoceros/gpgi/pull/265) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: enable ruff's TCH ruleset [#260](https://github.com/neutrinoceros/gpgi/pull/260) ([neutrinoceros](https://github.com/neutrinoceros))
- CLN: drop outdated comment [#259](https://github.com/neutrinoceros/gpgi/pull/259) ([neutrinoceros](https://github.com/neutrinoceros))
- API: clearly define public/private APIs [#256](https://github.com/neutrinoceros/gpgi/pull/256) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: add Python 3.13 to normal CI [#250](https://github.com/neutrinoceros/gpgi/pull/250) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: fixup coverage configuration [#248](https://github.com/neutrinoceros/gpgi/pull/248) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: avoid abusive uses of numpy.ones [#247](https://github.com/neutrinoceros/gpgi/pull/247) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: add `lock` parameter to `Dataset.deposit` [#246](https://github.com/neutrinoceros/gpgi/pull/246) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: simplify test dependency (coverage) [#245](https://github.com/neutrinoceros/gpgi/pull/245) ([neutrinoceros](https://github.com/neutrinoceros))
- PERF: release the GIL in hotloops [#244](https://github.com/neutrinoceros/gpgi/pull/244) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: set a 10min timeout [#243](https://github.com/neutrinoceros/gpgi/pull/243) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: never compile dependencies when testing [#238](https://github.com/neutrinoceros/gpgi/pull/238) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: run concurrency tests [#237](https://github.com/neutrinoceros/gpgi/pull/237) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: refactor concurrency tests to reduce duplication [#235](https://github.com/neutrinoceros/gpgi/pull/235) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: consistent use of the enum namespace [#234](https://github.com/neutrinoceros/gpgi/pull/234) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: make `Dataset._setup_host_cell_index` explicitly thread-safe [#233](https://github.com/neutrinoceros/gpgi/pull/233) ([neutrinoceros](https://github.com/neutrinoceros))
- DEP: drop support for CPython 3.10 and numpy<1.25 [#232](https://github.com/neutrinoceros/gpgi/pull/232) ([neutrinoceros](https://github.com/neutrinoceros))
- WHL: enable cp313 wheels [#231](https://github.com/neutrinoceros/gpgi/pull/231) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: fix type checking for the `method` argument in `Dataset.deposit` [#230](https://github.com/neutrinoceros/gpgi/pull/230) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: use Cython nightlies in bleeding edge tests [#227](https://github.com/neutrinoceros/gpgi/pull/227) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: use non-interactive mpl backend in tests [#224](https://github.com/neutrinoceros/gpgi/pull/224) ([neutrinoceros](https://github.com/neutrinoceros))
- API: forbid integer datatypes [#223](https://github.com/neutrinoceros/gpgi/pull/223) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: avoid creating a requirement file in minimal deps jobs [#222](https://github.com/neutrinoceros/gpgi/pull/222) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#220](https://github.com/neutrinoceros/gpgi/pull/220) ([pre-commit-ci](https://github.com/pre-commit-ci))
- ENH: implement `repr()` for `Grid`, `ParticleSet` and `Dataset` for introspection [#219](https://github.com/neutrinoceros/gpgi/pull/219) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: migrate from pip to uv in CI [#218](https://github.com/neutrinoceros/gpgi/pull/218) ([neutrinoceros](https://github.com/neutrinoceros))
- API: do not emit a warning when depositing on a single cell [#215](https://github.com/neutrinoceros/gpgi/pull/215) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: switch to uv in cibuildwheel [#214](https://github.com/neutrinoceros/gpgi/pull/214) ([neutrinoceros](https://github.com/neutrinoceros))
- API: forbid use of non-finite boxes [#213](https://github.com/neutrinoceros/gpgi/pull/213) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: refine auto-completion for grid.cell_edges and particles.coordinates [#212](https://github.com/neutrinoceros/gpgi/pull/212) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: rename `gpgi._IS_PYLIB` -> `gpgi._IS_PY_LIB` [#211](https://github.com/neutrinoceros/gpgi/pull/211) ([neutrinoceros](https://github.com/neutrinoceros))
- API: make `geometry` and `grid` mandatory keyword arguments to `gpgi.load` [#210](https://github.com/neutrinoceros/gpgi/pull/210) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: skip a test (non implemented feature) with GPGI_PY_LIB [#209](https://github.com/neutrinoceros/gpgi/pull/209) ([neutrinoceros](https://github.com/neutrinoceros))
- BUG: adjust inline assertions in GPGI_PY_LIB [#208](https://github.com/neutrinoceros/gpgi/pull/208) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: cleanup redundant call to pip list in CI and add info to pyttest logs [#207](https://github.com/neutrinoceros/gpgi/pull/207) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement GPGI_PY_LIB cleanup on Windows [#206](https://github.com/neutrinoceros/gpgi/pull/206) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: introduce `gpgi._IS_PYLIB` runtime constant [#205](https://github.com/neutrinoceros/gpgi/pull/205) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test GPGI_PY_LIB [#204](https://github.com/neutrinoceros/gpgi/pull/204) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: update .gitignore [#203](https://github.com/neutrinoceros/gpgi/pull/203) ([neutrinoceros](https://github.com/neutrinoceros))
- API: reduce size of wheels (hide unused but previously public API) [#202](https://github.com/neutrinoceros/gpgi/pull/202) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test against CPython 3.13 (GIL flavor) [#201](https://github.com/neutrinoceros/gpgi/pull/201) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: drop misleading use of typing.Protocol [#199](https://github.com/neutrinoceros/gpgi/pull/199) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test CPython 3.13 (free-threading flavor) [#198](https://github.com/neutrinoceros/gpgi/pull/198) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#195](https://github.com/neutrinoceros/gpgi/pull/195) ([pre-commit-ci](https://github.com/pre-commit-ci))
- REL: prepare release 1.0.0 [#174](https://github.com/neutrinoceros/gpgi/pull/174) ([neutrinoceros](https://github.com/neutrinoceros))

## [v1.0.0](https://github.com/neutrinoceros/gpgi/tree/v1.0.0) - 2024-03-30

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.14.0...v1.0.0)

### Fixed

- BUG: fix a spurious error when parsing invalid equatorial coordinates [#187](https://github.com/neutrinoceros/gpgi/pull/187) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: fix naming consistency in doc [#171](https://github.com/neutrinoceros/gpgi/pull/171) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: add missing documentation for sorting features [#168](https://github.com/neutrinoceros/gpgi/pull/168) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: add missing docstrings and fix linting errors in existing docstrings [#165](https://github.com/neutrinoceros/gpgi/pull/165) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: fix coverage checking [#150](https://github.com/neutrinoceros/gpgi/pull/150) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- TST: move --parallel-mode from configuration file to workflow to allow local coverage runs [#194](https://github.com/neutrinoceros/gpgi/pull/194) ([neutrinoceros](https://github.com/neutrinoceros))
- CLN: cleanup unused future import [#193](https://github.com/neutrinoceros/gpgi/pull/193) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test wheels on MacOS AMD [#192](https://github.com/neutrinoceros/gpgi/pull/192) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: manual upgrade for pre-commit [#191](https://github.com/neutrinoceros/gpgi/pull/191) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: clarify minimal deps test job [#190](https://github.com/neutrinoceros/gpgi/pull/190) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: use uv pip compile to keep minimal deps in sync [#189](https://github.com/neutrinoceros/gpgi/pull/189) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: refactor a test for readability [#188](https://github.com/neutrinoceros/gpgi/pull/188) ([neutrinoceros](https://github.com/neutrinoceros))
- RFC: refactor geometry validation with match/case [#186](https://github.com/neutrinoceros/gpgi/pull/186) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: drop support for CPython 3.9 and numpy<1.23 following SPEC 0 [#185](https://github.com/neutrinoceros/gpgi/pull/185) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: show ruff fixes in pre-commit logs [#184](https://github.com/neutrinoceros/gpgi/pull/184) ([neutrinoceros](https://github.com/neutrinoceros))
- DEPR: expire deprecations and bump version to 1.0.0dev0 [#182](https://github.com/neutrinoceros/gpgi/pull/182) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#179](https://github.com/neutrinoceros/gpgi/pull/179) ([pre-commit-ci](https://github.com/pre-commit-ci))
- MNT: group dependabot updates [#176](https://github.com/neutrinoceros/gpgi/pull/176) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: typecheck against Python 3.9 [#175](https://github.com/neutrinoceros/gpgi/pull/175) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: prepare release 1.0.0 [#174](https://github.com/neutrinoceros/gpgi/pull/174) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: fix typos [#173](https://github.com/neutrinoceros/gpgi/pull/173) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: opt-in numpy specific ruff linting rules [#170](https://github.com/neutrinoceros/gpgi/pull/170) ([neutrinoceros](https://github.com/neutrinoceros))
- STY: migrate from black to ruff-format [#167](https://github.com/neutrinoceros/gpgi/pull/167) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: update pre-commit configuration for compatibility with Python 3.12 [#166](https://github.com/neutrinoceros/gpgi/pull/166) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: use normalized issue prefix for auto reports [#163](https://github.com/neutrinoceros/gpgi/pull/163) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#159](https://github.com/neutrinoceros/gpgi/pull/159) ([pre-commit-ci](https://github.com/pre-commit-ci))
- TST: move tests to Python 3.12 (stable) [#157](https://github.com/neutrinoceros/gpgi/pull/157) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: don't skip tests for cp312 wheels [#156](https://github.com/neutrinoceros/gpgi/pull/156) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: simplify build dependencies [#155](https://github.com/neutrinoceros/gpgi/pull/155) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: cleanup redundant classifier [#154](https://github.com/neutrinoceros/gpgi/pull/154) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: switch to faster black pre-commit hook [#153](https://github.com/neutrinoceros/gpgi/pull/153) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: bump version to 0.14.0 [#151](https://github.com/neutrinoceros/gpgi/pull/151) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.14.0](https://github.com/neutrinoceros/gpgi/tree/v0.14.0) - 2023-09-04

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.13.0...v0.14.0)

### Added

- ENH: add sorting API [#149](https://github.com/neutrinoceros/gpgi/pull/149) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: add support for passing metadata to custom deposition methods [#122](https://github.com/neutrinoceros/gpgi/pull/122) ([neutrinoceros](https://github.com/neutrinoceros))

### Fixed

- BUG: fix error message so it's identical on numpy 1.x and 2.0 [#147](https://github.com/neutrinoceros/gpgi/pull/147) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- REL: bump version to 0.14.0 [#151](https://github.com/neutrinoceros/gpgi/pull/151) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: improvements to future-proofing [#146](https://github.com/neutrinoceros/gpgi/pull/146) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: test wheels as part of the build/release process [#145](https://github.com/neutrinoceros/gpgi/pull/145) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: add wheels for musllinux [#144](https://github.com/neutrinoceros/gpgi/pull/144) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: add wheels for CPython 3.12 [#143](https://github.com/neutrinoceros/gpgi/pull/143) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: run check_manifest in CI [#141](https://github.com/neutrinoceros/gpgi/pull/141) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: add a test to check compilation with build command [#140](https://github.com/neutrinoceros/gpgi/pull/140) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: unpin Cython [#138](https://github.com/neutrinoceros/gpgi/pull/138) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: drop support for Python 3.8 and numpy<1.21 [#136](https://github.com/neutrinoceros/gpgi/pull/136) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: add wheels for MacOS M1 arch [#135](https://github.com/neutrinoceros/gpgi/pull/135) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: migrate to Cython 3.0, forbid deprecated numpy API [#134](https://github.com/neutrinoceros/gpgi/pull/134) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: migrate to PyPI trusted-publishing [#133](https://github.com/neutrinoceros/gpgi/pull/133) ([neutrinoceros](https://github.com/neutrinoceros))
- DEP: set upper limit to runtime numpy [#132](https://github.com/neutrinoceros/gpgi/pull/132) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: migrate from oldest-supported-numpy to NPY_TARGET_VERSION [#131](https://github.com/neutrinoceros/gpgi/pull/131) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#130](https://github.com/neutrinoceros/gpgi/pull/130) ([pre-commit-ci](https://github.com/pre-commit-ci))
- TST: add future-proofing scheduled tests [#128](https://github.com/neutrinoceros/gpgi/pull/128) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: typecheck with Python 3.11 too [#127](https://github.com/neutrinoceros/gpgi/pull/127) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.13.0](https://github.com/neutrinoceros/gpgi/tree/v0.13.0) - 2023-06-11

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.12.0...v0.13.0)

### Added

- ENH: add support for passing metadata to custom deposition methods [#122](https://github.com/neutrinoceros/gpgi/pull/122) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: allow passing a callable as a deposition method [#121](https://github.com/neutrinoceros/gpgi/pull/121) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.12.0](https://github.com/neutrinoceros/gpgi/tree/v0.12.0) - 2023-06-09

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.11.2...v0.12.0)

### Added

- ENH: allow passing a callable as a deposition method [#121](https://github.com/neutrinoceros/gpgi/pull/121) ([neutrinoceros](https://github.com/neutrinoceros))

### Fixed

- BUG: fix a bug where ds.grid.cell_volumes would be of a different shape than ds.grid.shape [#114](https://github.com/neutrinoceros/gpgi/pull/114) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- MNT: cleanup unused lines in setup.py [#117](https://github.com/neutrinoceros/gpgi/pull/117) ([neutrinoceros](https://github.com/neutrinoceros))
- BLD: migrate to src layout [#116](https://github.com/neutrinoceros/gpgi/pull/116) ([neutrinoceros](https://github.com/neutrinoceros))
- STY: activate flake8-comprehensions and flake8-2020 (ruff) [#115](https://github.com/neutrinoceros/gpgi/pull/115) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.11.2](https://github.com/neutrinoceros/gpgi/tree/v0.11.2) - 2023-05-05

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.11.1...v0.11.2)

### Fixed

- BUG: fix a bug where ds.grid.cell_volumes would be of a different shape than ds.grid.shape [#114](https://github.com/neutrinoceros/gpgi/pull/114) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- REL: prep release 0.11.1 [#108](https://github.com/neutrinoceros/gpgi/pull/108) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.11.1](https://github.com/neutrinoceros/gpgi/tree/v0.11.1) - 2023-04-29

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.11.0...v0.11.1)

### Fixed

- BUG: resolve bugs with literal edge cases (particles living on the edge of the domain) [#107](https://github.com/neutrinoceros/gpgi/pull/107) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: a better headline [#106](https://github.com/neutrinoceros/gpgi/pull/106) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- REL: prep release 0.11.1 [#108](https://github.com/neutrinoceros/gpgi/pull/108) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#104](https://github.com/neutrinoceros/gpgi/pull/104) ([pre-commit-ci](https://github.com/pre-commit-ci))
- BLD: exclude generated C code from source distributions [#100](https://github.com/neutrinoceros/gpgi/pull/100) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: upgrade pre-commit hooks, simplify ruff config [#99](https://github.com/neutrinoceros/gpgi/pull/99) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: bump version to 0.11.0 [#95](https://github.com/neutrinoceros/gpgi/pull/95) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.11.0](https://github.com/neutrinoceros/gpgi/tree/v0.11.0) - 2023-02-09

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.10.1...v0.11.0)

### Fixed

- BUG: fix a bug where float32 coordinates were incorrectly invalidated against double precision limits [#91](https://github.com/neutrinoceros/gpgi/pull/91) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- REL: bump version to 0.11.0 [#95](https://github.com/neutrinoceros/gpgi/pull/95) ([neutrinoceros](https://github.com/neutrinoceros))
- UX: improve error messages in case of mixed dtypes [#94](https://github.com/neutrinoceros/gpgi/pull/94) ([neutrinoceros](https://github.com/neutrinoceros))
- UX: ensure that coordinates are formatted with consistent types (float) in error messages [#93](https://github.com/neutrinoceros/gpgi/pull/93) ([neutrinoceros](https://github.com/neutrinoceros))
- UX: improve error message in case of unexpected axes ordering [#92](https://github.com/neutrinoceros/gpgi/pull/92) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.10.1](https://github.com/neutrinoceros/gpgi/tree/v0.10.1) - 2023-02-08

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.10.0...v0.10.1)

### Other

- REL: prep release 0.10.0 [#90](https://github.com/neutrinoceros/gpgi/pull/90) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.10.0](https://github.com/neutrinoceros/gpgi/tree/v0.10.0) - 2023-02-08

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.9.0...v0.10.0)

### Other

- REL: prep release 0.10.0 [#90](https://github.com/neutrinoceros/gpgi/pull/90) ([neutrinoceros](https://github.com/neutrinoceros))
- API: invert definitions of CYLINDRICAL and POLAR geometries [#89](https://github.com/neutrinoceros/gpgi/pull/89) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: setup dependabot for github workflows [#85](https://github.com/neutrinoceros/gpgi/pull/85) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: prep release 0.9.0 [#84](https://github.com/neutrinoceros/gpgi/pull/84) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.9.0](https://github.com/neutrinoceros/gpgi/tree/v0.9.0) - 2023-02-08

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.8.0...v0.9.0)

### Other

- DOC: fix a typo and rephrase a sentence [#83](https://github.com/neutrinoceros/gpgi/pull/83) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: treat warnings as errors [#82](https://github.com/neutrinoceros/gpgi/pull/82) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: Dataset now respects the Liskov substitution principle  [#81](https://github.com/neutrinoceros/gpgi/pull/81) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: use recent version of numpy for typechecking [#79](https://github.com/neutrinoceros/gpgi/pull/79) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: pin numpy version used for typechecking [#78](https://github.com/neutrinoceros/gpgi/pull/78) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: setup dependabot [#76](https://github.com/neutrinoceros/gpgi/pull/76) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: move minimal_dependencies.txt to a separate directory [#75](https://github.com/neutrinoceros/gpgi/pull/75) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: prep release 0.8.0 [#73](https://github.com/neutrinoceros/gpgi/pull/73) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.8.0](https://github.com/neutrinoceros/gpgi/tree/v0.8.0) - 2023-02-03

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.7.2...v0.8.0)

### Fixed

- DOC: flesh out algorithmic details [#67](https://github.com/neutrinoceros/gpgi/pull/67) ([neutrinoceros](https://github.com/neutrinoceros))
- BUG: fix boundary recipe validation for functools.partial recipes [#66](https://github.com/neutrinoceros/gpgi/pull/66) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- BLD: declare explicit build backend (setuptools) [#72](https://github.com/neutrinoceros/gpgi/pull/72) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: use an StrEnum for geometry [#71](https://github.com/neutrinoceros/gpgi/pull/71) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: move test dependencies to requirement files [#70](https://github.com/neutrinoceros/gpgi/pull/70) ([neutrinoceros](https://github.com/neutrinoceros))
- STY: migrate linting to ruff, upgrade black [#69](https://github.com/neutrinoceros/gpgi/pull/69) ([neutrinoceros](https://github.com/neutrinoceros))
- [pre-commit.ci] pre-commit autoupdate [#68](https://github.com/neutrinoceros/gpgi/pull/68) ([pre-commit-ci](https://github.com/pre-commit-ci))

## [v0.7.2](https://github.com/neutrinoceros/gpgi/tree/v0.7.2) - 2022-11-17

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.7.1...v0.7.2)

### Fixed

- BUG: fix boundary recipe validation for functools.partial recipes [#66](https://github.com/neutrinoceros/gpgi/pull/66) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: fix a typo in documentation [#65](https://github.com/neutrinoceros/gpgi/pull/65) ([neutrinoceros](https://github.com/neutrinoceros))
- BUG: fix compatibility with numpy 1.18 [#64](https://github.com/neutrinoceros/gpgi/pull/64) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.7.1](https://github.com/neutrinoceros/gpgi/tree/v0.7.1) - 2022-11-15

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.7.0...v0.7.1)

### Fixed

- BUG: fix compatibility with numpy 1.18 [#64](https://github.com/neutrinoceros/gpgi/pull/64) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.7.0](https://github.com/neutrinoceros/gpgi/tree/v0.7.0) - 2022-11-06

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.6.0...v0.7.0)

### Added

- ENH: add support for weight fields [#59](https://github.com/neutrinoceros/gpgi/pull/59) ([neutrinoceros](https://github.com/neutrinoceros))

### Fixed

- BUG: fix a bug in boundary condition treatment (erroneous data selection) [#61](https://github.com/neutrinoceros/gpgi/pull/61) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- ENH: minimal support for Grid.cell_volumes property (cartesian only) [#60](https://github.com/neutrinoceros/gpgi/pull/60) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: improve type checking for builtin boundary recipes [#58](https://github.com/neutrinoceros/gpgi/pull/58) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: drop undocumented caching mechanism [#57](https://github.com/neutrinoceros/gpgi/pull/57) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.6.0](https://github.com/neutrinoceros/gpgi/tree/v0.6.0) - 2022-10-30

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.5.0...v0.6.0)

### Fixed

- BUG: fix missing stacklevel arg in a userwarning [#53](https://github.com/neutrinoceros/gpgi/pull/53) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- MNT: add support for Python 3.8 [#55](https://github.com/neutrinoceros/gpgi/pull/55) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: automate wheels publication to PyPI [#54](https://github.com/neutrinoceros/gpgi/pull/54) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.5.0](https://github.com/neutrinoceros/gpgi/tree/v0.5.0) - 2022-10-29

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.4.0...v0.5.0)

### Other

- ENH: implement extensible boundary condition API [#52](https://github.com/neutrinoceros/gpgi/pull/52) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: finish switching CI to Python 3.11 [#51](https://github.com/neutrinoceros/gpgi/pull/51) ([neutrinoceros](https://github.com/neutrinoceros))
- DEPR: rename 'pic' -> 'ngp', deprecate pic [#50](https://github.com/neutrinoceros/gpgi/pull/50) ([neutrinoceros](https://github.com/neutrinoceros))
- STY: add linters for yaml and cython [#49](https://github.com/neutrinoceros/gpgi/pull/49) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: flesh out test matrix [#48](https://github.com/neutrinoceros/gpgi/pull/48) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.4.0](https://github.com/neutrinoceros/gpgi/tree/v0.4.0) - 2022-10-19

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.3.0...v0.4.0)

### Other

- ENH: add support for storing arbitrary metadata in Dataset [#45](https://github.com/neutrinoceros/gpgi/pull/45) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: test that 2D projections of 3D deposit match 2D deposit of projections [#44](https://github.com/neutrinoceros/gpgi/pull/44) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: add type hints for deposition methods [#43](https://github.com/neutrinoceros/gpgi/pull/43) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement option to output ghost layers in Dataset.deposit [#42](https://github.com/neutrinoceros/gpgi/pull/42) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.3.0](https://github.com/neutrinoceros/gpgi/tree/v0.3.0) - 2022-10-08

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.2.0...v0.3.0)

### Other

- ENH: add equatorial geometry [#41](https://github.com/neutrinoceros/gpgi/pull/41) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement CIC deposition method [#40](https://github.com/neutrinoceros/gpgi/pull/40) ([neutrinoceros](https://github.com/neutrinoceros))
- TYP: add py.typed marker file to improve downstream type-checking [#39](https://github.com/neutrinoceros/gpgi/pull/39) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.2.0](https://github.com/neutrinoceros/gpgi/tree/v0.2.0) - 2022-10-07

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/v0.1.0...v0.2.0)

### Fixed

- BUG: fix a bug where field cache wasn't used in case a deposit field was queried with the full method name *and* the abbreviated name [#26](https://github.com/neutrinoceros/gpgi/pull/26) ([neutrinoceros](https://github.com/neutrinoceros))

### Other

- ENH: save memory allocation for particle coordinates [#38](https://github.com/neutrinoceros/gpgi/pull/38) ([neutrinoceros](https://github.com/neutrinoceros))
- DOC: polish and flesh out docs [#37](https://github.com/neutrinoceros/gpgi/pull/37) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement Triangular Shaped Cloud deposition method [#33](https://github.com/neutrinoceros/gpgi/pull/33) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: add verbose option to Dataset.deposit [#31](https://github.com/neutrinoceros/gpgi/pull/31) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: fix test job concurrency rules [#30](https://github.com/neutrinoceros/gpgi/pull/30) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement ghost layer padding [#29](https://github.com/neutrinoceros/gpgi/pull/29) ([neutrinoceros](https://github.com/neutrinoceros))
- TST: better testing for deposition on stretched grids [#28](https://github.com/neutrinoceros/gpgi/pull/28) ([neutrinoceros](https://github.com/neutrinoceros))
- MNT: fix image test reports [#27](https://github.com/neutrinoceros/gpgi/pull/27) ([neutrinoceros](https://github.com/neutrinoceros))
- ENH: implement fast particle indexing for constant stepping grids [#23](https://github.com/neutrinoceros/gpgi/pull/23) ([neutrinoceros](https://github.com/neutrinoceros))
- REL: v0.1.0 [#19](https://github.com/neutrinoceros/gpgi/pull/19) ([neutrinoceros](https://github.com/neutrinoceros))

## [v0.1.0](https://github.com/neutrinoceros/gpgi/tree/v0.1.0) - 2022-10-04

[Full Changelog](https://github.com/neutrinoceros/gpgi/compare/f8cfdbbbeec8d50b29979441522960452d1c4ddb...v0.1.0)
