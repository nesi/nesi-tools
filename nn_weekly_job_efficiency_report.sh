#!/bin/bash

if [[ "$0" == *"/"* ]]
then
	directory=$(/usr/bin/dirname $0)
else
	directory=$(/usr/bin/dirname $(/usr/bin/which $0))
fi

/usr/bin/which python3 >/dev/null 2>&1 || module load Python
/usr/bin/which python3 >/dev/null || exit 2

if [[ -t 0 ]]
then
	/usr/bin/echo "This script is intended for use as a cron job, rather than interactive use."
	/usr/bin/echo "To run an efficiency report interactively, use nn_seff_multi.py."
	exit 1
fi

timezone="Etc/UTC"
export TZ="${timezone}"

start=$(/usr/bin/date -d'last-tuesday - 1 week' +%Y-%m-%dT12:00:00)
end=$(/usr/bin/date -d'last-tuesday' +%Y-%m-%dT12:00:00)

memory=1536
if [[ "$1" == "maui" ]]
then
	memory=1152
fi
if [[ -z "$1" ]]
then
	python3 ${directory}/nn_seff_multi.py -S "${start}" -E "${end}" -m "${memory}"
else
	python3 ${directory}/nn_seff_multi.py -S "${start}" -E "${end}" -M "$1" -m "${memory}"
fi
