#!/usr/bin/env python3
# Copyright 2025 Google LLC

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

from typing import Any

class Operand():
    def __init__(self, json: dict) -> None:
        val = json.get('value',None)

        self._obj = json

    @property
    def enumerant(self) -> str:
        result = self._obj.get('enumerant', None)
        if result is None:
            raise Exception("operand needs an enumerant string")
        return result

    @property
    def value(self) -> int:
        val: str|int = self._obj['value']
        if isinstance(val, int):
            return val
        elif isinstance(val,str):
            if val.startswith("0x"):
                return int(val, 16)
            else:
                return int(val, 10)
        else:
            raise Exception("operand needs a value integer or string")

    @property
    def capabilities(self) -> list[str]:
        return self._obj.get('capabilities',[])

    @property
    def extensions(self) -> list[str]:
        return self._obj.get('extensions',[])

    @property
    def aliases(self) -> list[str]:
        return self._obj.get('aliases',[])

    @property
    def parameters(self) -> list[dict]:
        return self._obj.get('parameters',[])

    @property
    def version(self) -> str | None:
        return self._obj.get('version',None)

    @property
    def lastVersion(self) -> str | None:
        return self._obj.get('lastVersion',None)
