#!/bin/bash
echo "Building all submodules while checking out from MASTER branch."

git submodule update --init
git submodule foreach git checkout main
git submodule foreach git pull origin main
