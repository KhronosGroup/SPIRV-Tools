#!/usr/bin/env python

# Copyright 2016 The Shaderc Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Updates build-version.inc in the current directory, unless the update is
# identical to the existing content.
#
# Args: <spirv-tools_dir>
#
# For each directory, there will be a line in build-version.inc containing two
# strings:
#  - the software version deduced from the CHANGES file
#  - a longer string with the project name, the software version number, and
#    that directory's "git describe" output enclosed in double quotes and
#    appropriately escaped.

from __future__ import print_function

import datetime
import os.path
import re
import subprocess
import sys

OUTFILE = 'build-version.inc'


def command_output(cmd, dir):
    """Runs a command in a directory and returns its standard output stream.

    Captures the standard error stream.

    Raises a RuntimeError if the command fails to launch or otherwise fails.
    """
    p = subprocess.Popen(cmd,
                         cwd=dir,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (stdout, _) = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Failed to run %s in %s' % (cmd, dir))
    return stdout


def deduceSoftwareVersion(dir):
    """Returns a software version number parsed from the CHANGES file
    in the given dir.

    The CHANGES file describes most recent versions first.
    """

    pattern = re.compile('(v\d+\.\d+(wip)?) ')
    changes_file = os.path.join(dir, 'CHANGES')
    with open(changes_file) as f:
        for line in f.readlines():
            match = pattern.match(line)
            if match:
                return match.group(1)
    raise Exception('No version number found in {}'.format(changes_file))


def describe(dir):
    """Returns a string describing the current Git HEAD version as descriptively
    as possible.

    Runs 'git describe', or alternately 'git rev-parse HEAD', in dir.  If
    successful, returns the output; otherwise returns 'unknown hash, <date>'."""
    try:
        # decode() is needed here for Python3 compatibility. In Python2,
        # str and bytes are the same type, but not in Python3.
        # Popen.communicate() returns a bytes instance, which needs to be
        # decoded into text data first in Python3. And this decode() won't
        # hurt Python2.
        return command_output(['git', 'describe'], dir).rstrip().decode()
    except:
        try:
            return command_output(
                ['git', 'rev-parse', 'HEAD'], dir).rstrip().decode()
        except:
            return 'unknown hash, {}'.format(datetime.date.today().isoformat())


def main():
    if len(sys.argv) != 2:
        print('usage: {0} <spirv-tools_dir>'.format(sys.argv[0]))
        sys.exit(1)

    software_version = deduceSoftwareVersion(sys.argv[1])
    new_content = '"{}", "SPIRV-Tools {} {}\\n"\n'.format(
        software_version, software_version,
        describe(sys.argv[1]).replace('"', '\\"'))
    if os.path.isfile(OUTFILE) and new_content == open(OUTFILE, 'r').read():
        sys.exit(0)
    open(OUTFILE, 'w').write(new_content)

if __name__ == '__main__':
    main()
