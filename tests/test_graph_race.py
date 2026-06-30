"""验证 graph 连接竞态修复：推演中连接不可被其他 session 请求关闭。"""
import tempfile
import unittest
from pathlib import Path


class TestGraphRaceFix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge_test_")
        self.ws = Path(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_graph_keeps_running_session_alive(self):
        """推演中的图连接不被其他 session 的 get_graph 关闭。"""
        from strategy_forge.engine.engine import DeductionEngine

        engine = DeductionEngine(str(self.ws))

        # 通过 engine 创建两个 session
        s1 = engine.session_store.create("run001", "running", "test")
        s2 = engine.session_store.create("oth001", "other", "test")
        engine.session_store.update("run001", status="simulating")

        # running session 获取图连接
        g1 = engine.get_graph("run001")
        g1.upsert_entity("e1", "TestEntity", "Concept")
        self.assertFalse(g1._closed, "图连接在创建后不应被关闭")
        self.assertEqual(g1.count_entities(), 1)

        # 请求另一个 session 的图 — 不应关闭 g1
        g2 = engine.get_graph("oth001")
        self.assertFalse(g1._closed, "推演中的图连接被其他请求意外关闭")

        # g1 仍然可用
        g1.upsert_entity("e2", "Entity2", "Concept")
        self.assertEqual(g1.count_entities(), 2)

        # g2 独立可用
        self.assertFalse(g2._closed)
        g2.upsert_entity("x1", "OtherEntity", "Concept")
        self.assertEqual(g2.count_entities(), 1)

        g1.close()
        g2.close()
        engine.close_graph()

    def test_get_graph_closes_completed_session(self):
        """已完成的 session 的图连接应可被新请求关闭。"""
        from strategy_forge.engine.engine import DeductionEngine

        engine = DeductionEngine(str(self.ws))
        engine.session_store.create("done001", "done", "test")
        engine.session_store.update("done001", status="complete")

        g1 = engine.get_graph("done001")
        g1.upsert_entity("e1", "E1", "Concept")

        engine.session_store.create("new001", "new", "test")
        g2 = engine.get_graph("new001")

        # g1 应已被关闭（旧行为，因 session 非 running）
        self.assertTrue(g1._closed)

        g2.close()
        engine.close_graph()


if __name__ == "__main__":
    unittest.main()
