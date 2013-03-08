#!/bin/sh
VERSION=`cat version`
PREV_VERSION=${VERSION}
NEW_VERSION=$(( ${PREV_VERSION} + 1 ))

#
# Check that everything is commited
#


echo Previous version: ${PREV_VERSION}
echo New version: ${NEW_VERSION}

cvs tag -c v${NEW_VERSION}
CVS_RET=$?

if [ ${CVS_RET} -ne 0 ] ; then
#    echo ${CVS_RET}
    echo cvs tag failed.  Perhaps there are uncommited files.
    echo Please commit all files before creating a new version.
    exit 1
fi

#
# Update the current version in the version file
#
echo ${NEW_VERSION} > version

cvs commit -m "Bumped up version to ${NEW_VERSION}" version

#
# Tag the version file with the latest version
#
cvs tag -F v${NEW_VERSION} version
