from sciai.tools.sandbox import SandboxRunner


def test_timeout_kills_and_recovers():
    s = SandboxRunner(timeout=3)
    try:
        slow = s.run("sympy.integrate", {"expr": "exp(x**x**x)*sin(x**7+tan(x))", "var": "x"})
        assert not slow.ok and slow.timed_out
        ok = s.run("sympy.diff", {"expr": "x**2", "var": "x"})
        assert ok.ok and ok.value["result"]["value"] == "2*x"
    finally:
        s.close()


def test_network_is_disabled_in_worker(sandbox):
    from sciai.tools import sandbox as mod
    out = sandbox._call("run", "units.check_dimensions",
                        {"quantity": "3 meter", "expected_unit": "meter"}, 10)
    assert out.ok
    # The hardening function itself:
    import socket
    original = socket.socket
    try:
        mod._harden_worker()
        try:
            socket.socket()
            raised = False
        except PermissionError:
            raised = True
        assert raised
    finally:
        socket.socket = original


def test_parse_errors_come_back_as_errors(sandbox):
    out = sandbox.run("sympy.simplify", {"expr": "__import__('os').system('echo hi')"})
    assert not out.ok and "not allowed" in out.error
