"""Check ROS build routing and failure handling without ROS or a compiler."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BuildRosTest(unittest.TestCase):
    def test_build_routes_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            (root / 'build/CMakeCache.txt').write_text('standalone cache')
            (root / 'build.sh').write_text('exit 99\n')
            opt = root / 'opt/ros'
            for distro, version in [('noetic', 1), ('humble', 2), ('jazzy', 2)]:
                folder = opt / distro
                folder.mkdir(parents=True)
                (folder / 'setup.bash').write_text(f'export ROS_DISTRO={distro} ROS_VERSION={version}\n')
            macros = root / 'Macros.h'
            macros.write_text('#define EIGEN_WORLD_VERSION 3\n#define EIGEN_MAJOR_VERSION 4\n')
            for filename in ('build_ros1.sh', 'build_ros2.sh'):
                script = (ROOT / filename).read_text().replace('/opt/ros', str(opt))
                script = script.replace('/usr/include/eigen3/Eigen/src/Core/util/Macros.h', str(macros))
                (root / filename).write_text(script)
            import ast
            embedded = (ROOT / 'build_ros1.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
            ast.parse(embedded)
            (root / 'scripts').mkdir()
            (root / 'scripts/install_deps.sh').write_text('exit "${DEPS_STATUS:-0}"\n')
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            for command in ['rospack', 'ros2', 'colcon', 'catkin_make_isolated', 'sudo']:
                stub = bin_dir / command
                status = '99' if command == 'sudo' else ('${BUILD_STATUS:-0}' if command in ('colcon', 'catkin_make_isolated') else '0')
                stub.write_text('#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$CALL_LOG"\nexit "' + status + '"\n')
                stub.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('ROS_')}
            env.update(PATH=str(bin_dir) + ':' + os.environ['PATH'], CALL_LOG=str(root / 'calls'))
            def run(script, *args, **extra):
                return subprocess.run(['bash', str(root / script), *args], cwd=root,
                                      env={**env, **extra}, text=True, capture_output=True)
            cases = [('build_ros1.sh', [], 'noetic', 'catkin_make_isolated', 'colcon'),
                     ('build_ros2.sh', ['humble'], 'humble', 'colcon', 'catkin_make_isolated'),
                     ('build_ros2.sh', ['jazzy'], 'jazzy', 'colcon', 'catkin_make_isolated')]
            for script, args, distro, builder, other in cases:
                (root / 'calls').write_text('')
                result = run(script, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(root / 'build_ros' / distro / 'install'), result.stdout)
                calls = (root / 'calls').read_text()
                self.assertIn(builder, calls)
                self.assertNotIn(other, calls)
                self.assertNotEqual(run(script, *args, DEPS_STATUS='7').returncode, 0)
                self.assertEqual(calls, (root / 'calls').read_text())
                result = run(script, *args, BUILD_STATUS='8')
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Build complete.', result.stdout)
            self.assertEqual((root / 'build/CMakeCache.txt').read_text(), 'standalone cache')
            self.assertEqual(len(list((root / 'build').iterdir())), 1)
            self.assertNotEqual(run('build_ros1.sh', 'jazzy').returncode, 0)
            self.assertNotEqual(run('build_ros1.sh', ROS_DISTRO='jazzy').returncode, 0)
            self.assertNotEqual(run('build_ros2.sh', 'noetic').returncode, 0)
            self.assertNotEqual(run('build_ros2.sh', 'invalid').returncode, 0)
            self.assertNotEqual(run('build_ros2.sh', 'jazzy', 'humble').returncode, 0)
            self.assertNotEqual(run('build_ros2.sh').returncode, 0)
            self.assertEqual(run('build_ros2.sh', ROS_DISTRO='jazzy').returncode, 0)



if __name__ == '__main__':
    unittest.main()
