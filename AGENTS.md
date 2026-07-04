# BitonicPixelSorterForAE Agent Guide

## Scope

This repository is a standalone, self-contained After Effects SDK plug-in port.
It must reference only the Adobe SDK Examples tree and files inside
`BitonicPixelSorterForAE`; do not add dependencies on `Palf_Plugins`.

## Branch Policy

- Keep `develop` as the permanent normal development branch.
- Create `develop` from `master` if it is absent before feature work.
- Keep `master` for release-ready history unless the user explicitly requests a
  different release workflow.
- Do not rewrite history, force-push, squash, or rebase unless explicitly asked.

## Compatibility Rules

- Treat `PF_ParamDef.uu.id` values as persisted project data.
- Never renumber existing parameter IDs, never reuse retired IDs, and append new
  IDs before `BPS_NUM_PARAMS`.
- UI order may differ from persisted ID order; maintain a separate checkout/UI
  index enum when moving controls.
- Do not change `MATCHNAME`, PiPL identifiers, or persisted data formats without
  explicit confirmation.

## Initially Hidden Parameters

Mode-dependent controls (Angle, Centre, Swirl, Path, and similar) must start
hidden when Mode defaults to Axis. Follow the shared skill — do not improvise:

- [ae-initially-hidden-params](../.agents/skills/ae-initially-hidden-params/SKILL.md)

Summary:

1. **ParamsSetup (creation only):** `PF_PUI_INVISIBLE` plus
   `PF_ParamFlag_COLLAPSE_TWIRLY` on `PF_ADD_ANGLE` / `PF_ADD_POINT` (and any
   control with a separate topic vs control region).
2. **Runtime:** `AEGP_DynStreamFlag_HIDDEN` only (`BPS_UpdateParamsUI`). Never
   toggle `PF_PUI_INVISIBLE` via `PF_UpdateParamUI` on After Effects.
3. **First paint:** call the same visibility update from `PF_Cmd_SEQUENCE_SETUP`
   and `PF_Cmd_SEQUENCE_RESETUP`, not only `UPDATE_PARAMS_UI`.

Removing `COLLAPSE_TWIRLY` from an initially-hidden ANGLE to “show the dial when
revealed” reintroduces the orphan dial on first apply. Keep `COLLAPSE_TWIRLY`.

## Build And Validation

- Prefer Windows Debug builds for development validation:
  `cmake --build build\Win --config Debug`.
- Do not update tracked Release deliverables under `dist/Win/Release` or
  `dist/Mac/Release` unless the user explicitly asks for packaging.
- When changing GPU parameters, keep CUDA, OpenCL, DirectX/HLSL, Metal, and host
  upload structs in the same field order.
- Record every agent-authored project-change session in
  `../.agents/production-record.md` from the workspace root.
