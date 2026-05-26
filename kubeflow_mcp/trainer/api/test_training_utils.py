import pytest
import textwrap

class TestMakeTrainFunc:
    
    def test_wrapped_script_executes(self):
        from kubeflow_mcp.trainer.api.training import _make_train_func
        import builtins
        import textwrap

        builtins._marker = []

        script = textwrap.dedent("""
    import builtins
    builtins._marker.append("ran")
    """)

        func = _make_train_func(script)
        func()

        assert builtins._marker == ["ran"]

    def test_async_train_executes(self):
        from kubeflow_mcp.trainer.api.training import _make_train_func
        import builtins
        import asyncio
        import textwrap

        builtins._marker = []

        script = textwrap.dedent("""
            import builtins

            async def train():
                builtins._marker.append("ran")
        """)

        func = _make_train_func(script)

        asyncio.run(func())

        assert builtins._marker == ["ran"]


    def test_func_args_are_used(self):
        from kubeflow_mcp.trainer.api.training import _make_train_func
        import builtins
        import textwrap
        builtins._marker = []

        script =textwrap.dedent("""
    import builtins

    def train(lr=None):
        builtins._marker.append(lr)
    """)

        func = _make_train_func(script, {"lr": 0.1})
        func(lr=0.1)

        assert builtins._marker == [0.1]

    def test_plain_script_with_args(self):
        from kubeflow_mcp.trainer.api.training import _make_train_func
        import builtins
        import textwrap

        builtins._marker = []

        script = textwrap.dedent("""
    import builtins
    builtins._marker.append("wrapped")
    """)

        func = _make_train_func(script, {"lr": 0.1})
        func(lr=0.1)

        assert builtins._marker == ["wrapped"]

    def test_missing_params_raises_value_error(self):
        from kubeflow_mcp.trainer.api.training import _make_train_func
        import textwrap

        script = textwrap.dedent("""
            def train():
                print("training")
        """)

        with pytest.raises(ValueError, match="missing required params"):
            _make_train_func(script, {"lr": 0.01})