"""Offline checks: mocked Firebase/Ollama, no accounts created or model calls."""
import json
import math
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
from streamlit.testing.v1 import AppTest
import firebase_client as auth
import rag

ROOT = Path(__file__).resolve().parents[1]
USER = {'uid': 'test-user', 'email': 'test@example.com', 'id_token': 'test-token',
        'expires_at': time.time() + 3600}


class AuthTests(unittest.TestCase):
    def test_expired_session(self):
        for user in (None, {}, dict(USER, expires_at=0), dict(USER, expires_at=math.nan)):
            with self.assertRaises(auth.FirebaseError) as error:
                auth.require_session(user)
            self.assertTrue(error.exception.sign_in_again)
        auth.require_session(USER)

    @patch.object(auth, 'FIREBASE_API_KEY', 'test-key')
    @patch.object(auth.requests, 'post')
    def test_login_and_no_password_retained(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {
            'localId': 'test-user', 'email': 'test@example.com',
            'idToken': 'test-token', 'expiresIn': '3600'}
        user = auth.sign_in(' test@example.com ', 'password')
        self.assertEqual(user['uid'], 'test-user')
        self.assertNotIn('password', user)
        self.assertIn('accounts:signInWithPassword', post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs['json']['email'], 'test@example.com')

    @patch.object(auth, 'FIREBASE_API_KEY', 'test-key')
    @patch.object(auth.requests, 'post')
    def test_invalid_login_and_network_error(self, post):
        post.return_value = Mock(ok=False, status_code=400)
        post.return_value.json.return_value = {'error': {'message': 'INVALID_LOGIN_CREDENTIALS'}}
        with self.assertRaisesRegex(auth.FirebaseError, 'Sign-in failed'):
            auth.sign_in('test@example.com', 'wrong')
        post.side_effect = requests.ConnectionError('private-key')
        with self.assertRaisesRegex(auth.FirebaseError, 'Cannot reach') as error:
            auth.sign_in('test@example.com', 'wrong')
        self.assertNotIn('private-key', str(error.exception))


class RagTests(unittest.TestCase):
    def setUp(self):
        self.content = (ROOT / 'knowledge_base.json').read_text()
        self.entries = rag.parse_guide(self.content)
        rag.build_index.clear()

    def test_guide_and_duplicate_ids(self):
        self.assertEqual(len(self.entries), 18)
        self.entries[1]['id'] = self.entries[0]['id']
        with self.assertRaises(rag.RagError):
            rag.parse_guide(json.dumps(self.entries))

    def test_invalid_embeddings(self):
        for vector in ([], [0, 0], [float('nan')], ['x'], [float('inf')]):
            with self.assertRaises(rag.RagError):
                rag._unit_vector(vector)

    @patch.object(rag, 'embed', return_value=((1.0, 0.0),))
    def test_cosine_ranking(self, embed):
        hits = rag.retrieve('question', self.entries[:3],
                            ((0, 1), (1, 0), (-1, 0)), 'http://localhost', 'model', k=2)
        self.assertEqual([e['id'] for e in hits], ['G02', 'G01'])
        self.assertEqual(hits[0]['score'], 1)

    @patch.object(rag, 'embed')
    def test_cache_reuse_and_data_invalidation(self, embed):
        embed.return_value = tuple((1.0, 0.0) for _ in self.entries)
        rag.build_index(self.content, 'http://localhost', 'model')
        rag.build_index(self.content, 'http://localhost', 'model')
        self.assertEqual(embed.call_count, 1)
        changed = self.entries.copy()
        changed[0] = dict(changed[0], text='Changed guide text.')
        rag.build_index(json.dumps(changed), 'http://localhost', 'model')
        self.assertEqual(embed.call_count, 2)
        rag.build_index(self.content, 'http://other-server', 'model')
        self.assertEqual(embed.call_count, 3)

    def test_citations_and_quotes(self):
        hits = [self.entries[0]]
        result = {'answerable': True, 'answer': 'It is a religious landmark. [G01]',
                  'evidence': [{'id': 'G01', 'quote': 'a major religious landmark in Venice'}]}
        self.assertEqual(rag.validated_answer(json.dumps(result), hits), result['answer'])
        result['answer'] = 'A claim. [G99]'
        self.assertEqual(rag.validated_answer(json.dumps(result), hits), rag.UNKNOWN)
        result['answer'] = 'A claim. [G01]'
        result['evidence'][0]['quote'] = 'Tickets cost 500 euros.'
        self.assertEqual(rag.validated_answer(json.dumps(result), hits), rag.UNKNOWN)
        self.assertEqual(rag.validated_answer('{broken', hits), rag.UNKNOWN)
        self.assertEqual(rag.validated_answer('{"answerable": false}', hits), rag.UNKNOWN)

    @patch.object(rag, 'OLLAMA_BASE_URL', 'http://localhost:11434')
    @patch.object(rag, '_post')
    def test_end_to_end_mocked_rag(self, post):
        quote = 'The Glass Museum on Murano, an island in Venice'
        vectors = [[0, 1] for _ in self.entries]
        vectors[4] = [1, 0]
        post.side_effect = [
            {'embeddings': vectors}, {'embeddings': [[1, 0]]},
            {'message': {'content': json.dumps({
                'answerable': True, 'answer': 'The Glass Museum is on Murano.',
                'evidence': [{'id': 'G05', 'quote': quote}]})}}]
        result = rag.answer_question('Where is the Glass Museum?')
        self.assertEqual(result['sources'][0]['id'], 'G05')
        self.assertEqual(result['answer'], 'The Glass Museum is on Murano. [G05]')
        self.assertEqual(post.call_args.args[1], '/api/chat')
        self.assertEqual(post.call_args.args[2]['model'], 'qwen3:4b')
        schema = post.call_args.args[2]['format']
        self.assertEqual(set(schema['required']), {'answerable', 'answer', 'evidence'})
        self.assertFalse(schema['additionalProperties'])

    @patch.object(rag, 'OLLAMA_BASE_URL', 'http://localhost:11434')
    @patch.object(rag, '_post')
    def test_low_similarity_skips_generation(self, post):
        post.side_effect = [{'embeddings': [[1, 0] for _ in self.entries]},
                            {'embeddings': [[0, 1]]}]
        self.assertEqual(rag.answer_question('Unrelated question')['answer'], rag.UNKNOWN)
        self.assertEqual(post.call_count, 2)
        with self.assertRaisesRegex(rag.RagError, 'enter a question'):
            rag.answer_question('   ')

    @patch.object(rag.requests, 'post')
    def test_server_failure(self, post):
        post.side_effect = requests.Timeout()
        with self.assertRaisesRegex(rag.RagError, 'timed out'):
            rag._post('http://localhost:11434', '/api/embed', {})
        post.side_effect = None
        post.return_value = Mock(ok=True)
        post.return_value.json.side_effect = ValueError()
        with self.assertRaisesRegex(rag.RagError, 'did not return JSON'):
            rag._post('http://localhost:11434', '/api/embed', {})


class InterfaceTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=10)

    @patch.object(auth, 'FIREBASE_API_KEY', 'test-key')
    def test_auth_gate_and_signout(self):
        app = self.app().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_input), 0)
        app.session_state['user'] = dict(USER)
        app.session_state['messages'] = [{'role': 'user', 'content': 'private history'}]
        app.run()
        self.assertEqual(len(app.chat_input), 1)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertNotIn('user', app.session_state)
        self.assertNotIn('messages', app.session_state)
        self.assertEqual(len(app.chat_input), 0)

    @patch.object(auth, 'FIREBASE_API_KEY', 'test-key')
    def test_expiry_clears_history(self):
        app = self.app()
        app.session_state['user'] = dict(USER, expires_at=0)
        app.session_state['messages'] = [{'role': 'user', 'content': 'private history'}]
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn('messages', app.session_state)
        self.assertEqual(len(app.chat_input), 0)

    @patch.object(auth, 'FIREBASE_API_KEY', 'test-key')
    @patch.object(rag, 'answer_question')
    def test_chat_and_server_error(self, answer):
        app = self.app()
        app.session_state['user'] = dict(USER)
        app.run()
        answer.return_value = {'answer': 'Murano. [G05]', 'sources': []}
        app.chat_input[0].set_value('Where is the Glass Museum?').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['messages']), 2)
        answer.side_effect = rag.RagError('Cannot reach Ollama.')
        app.chat_input[0].set_value('Another question').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['messages']), 2)
        self.assertEqual(app.error[0].value, 'Cannot reach Ollama.')


if __name__ == '__main__':
    unittest.main()
