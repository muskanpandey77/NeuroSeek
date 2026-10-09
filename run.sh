#!/bin/bash
cd "$(dirname "$0")"
javac -encoding UTF-8 Main.java && java Main
