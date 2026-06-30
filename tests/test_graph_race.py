"""验证 graph 连接竞态修复：推演中连接不可被其他 session 请求关闭。"""
import os
import tempfile
import unittest
import uuid
from pathlib import Path


class TestGraphRaceFix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge_test_")
        # 强制使用隔离的 data 目录
        os.environ["FORGE_DATA_DIR"] = str(Path(self.tmp) / ".data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_graph_keeps_running_session_alive(self):
        """推演中的图连接不被其他 session 的 get_graph 关闭。"""
        from strategy_forge.engine.engine import DeductionEngine

        engine = DeductionEngine(str(self.tmp))
        rid1 = uuid.uuid4().hex[:8]
        rid2 = uuid.uuid4().hex[:8]

        engine.session_store.create(rid1, "running", "test")
        engine.session_store.update(rid1, status="simulating")
        g1 = engine.get_graph(rid1)
        g1.upsert_entity("e1", "TestEntity", "Concept")
        self.assertFalse(g1._closed, "图连接在创建后不应被关闭")

        engine.session_store.create(rid2, "other", "test")
        g2 = engine.get_graph(rid2)
        self.assertFalse(g1._closed, "推演中的图连接被其他请求意外关闭")
        self.assertFalse(g2._closed)

        # g1 仍然可用
        g1.upsert_entity("e2", "Entity2", "Concept")
        self.assertEqual(g1.count_entities(), 2)

        g2.upsert_entity("x1", "Other", "Concept")
        self.assertEqual(g2.count_entities(), 1)

        g1.close()
        g2.close()
        engine.close_graph()

    def test_get_graph_closes_completed_session(self):
        """已完成的 session 的图连接应可被新请求关闭。"""
        from strategy_forge.engine.engine import DeductionEngine

        engine = DeductionEngine(str(self.tmp))
        rid1 = uuid.uuid4().hex[:8]
        rid2 = uuid.uuid4().hex[:8]

        engine.session_store.create(rid1, "done", "test")
        engine.session_store.update(rid1, status="complete")
        g1 = engine.get_graph(rid1)
        g1.upsert_entity("e1", "E1", "Concept")

        engine.session_store.create(rid2, "new", "test")
        g2 = engine.get_graph(rid2)
        self.assertTrue(g1._closed)

        g2.close()
        engine.close_graph()


if __name__ == "__main__":
    unittest.main()
