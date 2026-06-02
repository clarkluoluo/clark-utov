#!/usr/bin/env bash
# Launcher for the test runner (Sha256TestRunner). We DO NOT ship unidbg;
# user must provide it via $UNIDBG_HOME (a directory containing the
# unidbg-android jar + its transitive dependency jars).
#
# Typical $UNIDBG_HOME contents on a Maven user's machine:
#   $UNIDBG_HOME/
#     ├── unidbg-android-0.9.9.jar
#     ├── unidbg-api-0.9.9.jar
#     ├── unicorn-1.0.15.jar
#     ├── capstone-3.1.8.jar
#     ├── keystone-0.9.7.jar
#     ├── fastjson-1.2.83.jar
#     ├── jna-5.10.0.jar
#     ├── slf4j-api-2.0.16.jar
#     ├── apk-parser-2.6.10.jar
#     ├── native-lib-loader-2.3.5.jar
#     ├── commons-codec-1.21.0.jar
#     ├── commons-io-2.21.0.jar
#     ├── commons-collections4-4.5.0.jar
#     └── demumble-1.0.4.jar
#
# Cheap way to populate it (any Maven user):
#   mvn dependency:copy-dependencies \
#       -DincludeArtifactIds=unidbg-android \
#       -DoutputDirectory=$HOME/.unidbg/0.9.9
#   export UNIDBG_HOME=$HOME/.unidbg/0.9.9
#
# Usage:
#   ./bin/run-runner.sh demo                                   # one-shot test
#   ./bin/run-runner.sh serve <path/to/lib.so>                 # NDJSON server
#   ./bin/run-runner.sh serve <path/to/lib.so> </dev/stdin >...# wire to agent

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
RUNNER_JAR="${RUNNER_JAR:-$REPO_ROOT/example/runner-sha256/runner/target/sha256-test-runner-0.1.0.jar}"

if [[ -z "${UNIDBG_HOME:-}" ]]; then
    cat >&2 <<EOF
error: UNIDBG_HOME is not set.

This project does NOT bundle unidbg. You must supply it yourself.

Quick setup (if you have Maven):
    mkdir -p ~/.unidbg/0.9.9
    cd ~/.unidbg/0.9.9
    cat > pom.xml <<'POM'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local.unidbg</groupId><artifactId>collect</artifactId><version>1</version>
  <repositories>
    <repository><id>jitpack.io</id><url>https://jitpack.io</url></repository>
  </repositories>
  <dependencies>
    <dependency>
      <groupId>com.github.zhkl0228</groupId>
      <artifactId>unidbg-android</artifactId>
      <version>0.9.9</version>
    </dependency>
  </dependencies>
</project>
POM
    mvn dependency:copy-dependencies -DoutputDirectory=. -DincludeScope=runtime
    rm pom.xml
    export UNIDBG_HOME=\$HOME/.unidbg/0.9.9

Then re-run this command.

See DEPENDENCIES.md §unidbg for the version compatibility matrix.
EOF
    exit 2
fi

if [[ ! -d "$UNIDBG_HOME" ]]; then
    echo "error: UNIDBG_HOME=$UNIDBG_HOME does not exist or is not a directory" >&2
    exit 2
fi

if ! ls "$UNIDBG_HOME"/unidbg-android-*.jar >/dev/null 2>&1; then
    echo "error: no unidbg-android-*.jar found in $UNIDBG_HOME" >&2
    echo "       (expected the jar plus its transitive deps in this dir)" >&2
    exit 2
fi

if [[ ! -f "$RUNNER_JAR" ]]; then
    echo "error: runner jar not found at $RUNNER_JAR" >&2
    echo "       build it first: (cd example/runner-sha256/runner && mvn -DskipTests package)" >&2
    exit 2
fi

# Detect unidbg version from jar filename (e.g. unidbg-android-0.9.9.jar)
UNIDBG_VERSION="${UNIDBG_VERSION:-}"
if [[ -z "$UNIDBG_VERSION" ]]; then
    UNIDBG_JAR=$(ls "$UNIDBG_HOME"/unidbg-android-*.jar 2>/dev/null | head -1)
    if [[ -n "$UNIDBG_JAR" ]]; then
        UNIDBG_VERSION=$(basename "$UNIDBG_JAR" .jar | sed 's/^unidbg-android-//')
    fi
fi
export UNIDBG_VERSION

# Pass the version to the runner via system property so metadata() can include it.
exec java \
    -Dunidbg.home="$UNIDBG_HOME" \
    -Dunidbg.version="$UNIDBG_VERSION" \
    -cp "$UNIDBG_HOME/*:$RUNNER_JAR" \
    com.clarkutov.runner.test.Main "$@"
