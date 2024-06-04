import os
import subprocess

import garnet.workflow


def test_template(tmpdir):
    script = os.path.abspath("./src/garnet/workflow.py")
    command = ["python", script]

    for inst in garnet.workflow.inst_dict.keys():
        fname = inst + ".yaml"
        filepath = os.path.join(tmpdir, fname)
        subprocess.run(command + [filepath, "temp", inst], check=False)

        assert os.path.exists(filepath)
