# This file should be in /etc/profile.d or something like it, and
# it should be sourced after or from within /etc/bash.bashrc.local
# and friends.
# TODO: A C shell compatible version should also be prepared.
#
# The scripts provides system variable strings to distinguish between the different systems and software stacks.
# The top level modules will load the related software stack, e.g. from /opt/nesi/<system_arch_string>
#
# author: Mandes Schönherr mandes.schoenherr@nesi.org.nz

# Root does not need modules
if [ "`id -u`" -ne 0 ]; then
	# Derive operating system name and revision
	os_name=$(egrep "^NAME=" /etc/os-release | sed -e 's/^NAME="//g; s/"$//g; s/"$//g; s/ Linux//g' | tr '[:upper:]' '[:lower:]')
	os_version=$(egrep "^VERSION_ID=" /etc/os-release | sed -e 's/^VERSION_ID="//; s/"$//g' | cut -d"." -f1)

	# Derive processor type
	# Assume that all CPUs within the machine are the same type
	proctype_raw=$(grep "^model name" /proc/cpuinfo | sort | uniq | head -n 1)
	proctype="UNKNOWN"

	# destinguish the different NeSI systems
	case "${os_name}${os_version}_${proctype_raw}" in 
	centos7*Intel*Xeon*6148*)
		proctype="skl"
		system="CS500"
		;;
	centos7*Xeon*CPU*E5-2695*)
		proctype="bdw"
		system="CS400"
		;;
	centos7*Xeon*CPU*E7-4850*)
		proctype="bdw"
		system="CS400"
		;;
	sles12*Intel*Xeon*6148*)
		proctype="skl"
		system="XC50"
		;;
	sles12*Xeon*CPU*E5-2695*)
		proctype="skl"
		system="XC40"
		;;
	*)
		proctype="UNKNOWN"
		system="UNKNOWN"
		echo "This system architecture is not covered by the top level modules yet. Please report to support."
		;;
	esac
	# Provide the specification for the top level modules
	export OS_ARCH_STRING="${os_name}${os_version}"
	export CPUARCH_STRING="${proctype}"
	export SYSTEM_STRING="${system}"
	#echo ${SYSTEM_STRING}_${OS_ARCH_STRING}_${CPUARCH_STRING}
	
	# if not initialized, init the modules
	if [ -z "$MODULE_VERSION" ]; then
		if [ -e /opt/cray/pe/modules/default/etc/modules.sh ]; then
			. /opt/cray/pe/modules/default/etc/modules.sh
		elif [ -e /opt/modules/default/etc/modules.sh ]; then
			. /opt/modules/default/etc/modules.sh
		elif [ -e /cm/local/apps/environment-modules/current/init/bash ]; then
			. /cm/local/apps/environment-modules/current/init/bash
		else
			echo "ERROR: could not find modules tool for NeSI top level module initialization"
		fi
	fi
	# add path to the top level modules
	export MODULEPATH=/opt/nesi/modulefiles:$MODULEPATH
	# load the NeSI top level module
	module load NeSI
fi
