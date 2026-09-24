# Build and Submodule Decision Rules

For each build file check both fork additions and upstream changes:

- extension source lists and compile definitions;
- include/library directories and device guards;
- plugin package discovery and package data;
- version source and wheel metadata;
- editable install and isolated wheel behavior;
- optional dependencies and import-time loading.

For each gitlink choose upstream, fork, or a reviewed third value. Record why. A missing target gitlink can mean removal, not an error; distinguish it from a checkout failure.
