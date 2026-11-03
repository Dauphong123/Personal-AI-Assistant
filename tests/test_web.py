"""Web tools: URL safety, fetching, reading, searching. No network is used."""

import unittest
from unittest.mock import patch

from requests.structures import CaseInsensitiveDict

from tests import stubs

stubs.install()

import tools  # noqa: E402
import web  # noqa: E402

HOSTS = {  # what the fake DNS knows
    "example.com": ["93.184.216.34"],
    "docs.example.org": ["93.184.216.35"],
    "localhost": ["127.0.0.1", "::1"],
    "intranet.corp": ["10.0.0.5"],
    "evil.example": ["93.184.216.99"],
    "mapped.example": ["::ffff:127.0.0.1"],
    "metadata.example": ["169.254.169.254"],
}


def fake_resolve(host, port):
    if host in HOSTS:
        return HOSTS[host]

    try:  # an IP literal resolves to itself
        import ipaddress

        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        raise OSError("unknown host") from None


class FakeResponse:
    def __init__(self, body=b"", status=200, content_type="text/html", location=None):
        self.status_code = status
        self.headers = CaseInsensitiveDict({"Content-Type": content_type})

        if location:
            self.headers["Location"] = location

        self._body = body
        self.closed = False

    def iter_content(self, size):
        for start in range(0, len(self._body), size):
            yield self._body[start : start + size]

    def close(self):
        self.closed = True


def page(body: str, **kwargs) -> FakeResponse:
    return FakeResponse(body.encode("utf-8"), **kwargs)


class WebTestCase(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(web, "_resolve", fake_resolve)
        patcher.start()
        self.addCleanup(patcher.stop)
        web._pages._items.clear()
        web._searches._items.clear()
        self.requested = []

    def serve(self, routes):
        """Answer http_get from {url: FakeResponse}; unknown URLs are a test failure."""

        def fake_get(url):
            self.requested.append(url)
            return routes[url]

        patcher = patch.object(web, "http_get", fake_get)
        patcher.start()
        self.addCleanup(patcher.stop)


class SettingsAndCacheTests(unittest.TestCase):
    def test_web_access_can_be_turned_off(self):
        self.assertTrue(web.web_enabled({}))
        self.assertTrue(web.web_enabled({"WEB_SEARCH_ENABLED": "true"}))

        for off in ["false", "0", "no", "OFF"]:
            self.assertFalse(web.web_enabled({"WEB_SEARCH_ENABLED": off}), off)

    def test_cache_entries_expire(self):
        now = [0.0]
        cache = web.TTLCache(ttl=10, size=5, clock=lambda: now[0])
        cache.put("a", 1)
        self.assertEqual(cache.get("a"), 1)
        now[0] = 11
        self.assertIsNone(cache.get("a"))

    def test_cache_drops_the_oldest_when_full(self):
        now = [0.0]
        cache = web.TTLCache(ttl=100, size=2, clock=lambda: now[0])
        cache.put("a", 1)
        now[0] = 1
        cache.put("b", 2)
        now[0] = 2
        cache.put("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual((cache.get("b"), cache.get("c")), (2, 3))


class CheckUrlTests(WebTestCase):
    def test_public_pages_are_allowed(self):
        for url in ["https://example.com/a?b=1", "http://docs.example.org", "http://93.184.216.34/x"]:
            self.assertIsNone(web.check_url(url), url)

    def test_internal_addresses_are_blocked(self):
        for url in [
            "http://localhost:8000", "http://127.0.0.1", "http://10.1.2.3", "http://192.168.1.1/admin",
            "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://0.0.0.0",
            "http://intranet.corp/", "http://metadata.example/", "http://mapped.example/",
            "http://172.16.0.1", "http://100.64.0.1",
        ]:  # fmt: skip
            self.assertIn("blocked", web.check_url(url) or "", url)

    def test_other_schemes_and_credentials_are_refused(self):
        self.assertIn("http and https", web.check_url("file:///etc/passwd"))
        self.assertIn("http and https", web.check_url("ftp://example.com/x"))
        self.assertIn("http and https", web.check_url("javascript:alert(1)"))
        self.assertIn("user name", web.check_url("http://user:pass@example.com"))
        self.assertIn("no host", web.check_url("http:///path"))

    def test_unknown_hosts_are_reported(self):
        self.assertIn("could not find", web.check_url("http://nope.invalid"))

    def test_a_name_with_one_private_address_is_blocked(self):
        with patch.dict(HOSTS, {"mixed.example": ["93.184.216.34", "10.0.0.1"]}):
            self.assertIn("blocked", web.check_url("http://mixed.example"))


class DnsRebindingTests(WebTestCase):
    """The connection must use the addresses that were checked, not a second DNS answer."""

    def test_lookups_inside_a_request_are_answered_from_the_checked_addresses(self):
        import socket

        attacker_answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

        with patch.object(web, "_real_getaddrinfo", return_value=attacker_answer):
            self.assertEqual(socket.getaddrinfo("example.com", 80)[0][4][0], "127.0.0.1")

            with web._pinned("example.com", ["93.184.216.34"]):
                found = socket.getaddrinfo("example.com", 443)
                self.assertEqual([f[4] for f in found], [("93.184.216.34", 443)])
                self.assertEqual(socket.getaddrinfo("other.example", 80)[0][4][0], "127.0.0.1", "other hosts are untouched")

            self.assertEqual(socket.getaddrinfo("example.com", 80)[0][4][0], "127.0.0.1", "released afterwards")

    def test_a_rebinding_dns_server_cannot_redirect_the_connection(self):
        import socket

        answers = iter([["93.184.216.34"], ["127.0.0.1"], ["127.0.0.1"]])  # public for the check, then evil
        connected_to = []

        def resolve(host, port):
            return next(answers)

        def http_get(url):
            connected_to.append(socket.getaddrinfo("rebind.example", 80)[0][4][0])
            return page("<p>hello</p>")

        with patch.object(web, "_resolve", resolve), patch.object(web, "http_get", http_get):
            web.fetch("http://rebind.example/")

        self.assertEqual(connected_to, ["93.184.216.34"])

    def test_every_redirect_hop_is_checked_and_pinned_separately(self):
        import socket

        resolved = []
        connected = []
        routes = {
            "http://a.example/": page("", status=302, location="http://b.example/"),
            "http://b.example/": page("<p>ok</p>"),
        }
        table = {"a.example": ["93.184.216.40"], "b.example": ["93.184.216.41"]}

        def resolve(host, port):
            resolved.append(host)
            return table[host]

        def http_get(url):
            host = url.split("/")[2]
            connected.append((host, socket.getaddrinfo(host, 80)[0][4][0]))
            return routes[url]

        with patch.object(web, "_resolve", resolve), patch.object(web, "http_get", http_get):
            web.fetch("http://a.example/")

        self.assertEqual(resolved, ["a.example", "b.example"], "one lookup per hop")
        self.assertEqual(connected, [("a.example", "93.184.216.40"), ("b.example", "93.184.216.41")])

    def test_pins_are_released_even_when_the_request_fails(self):
        import requests

        def boom(url):
            raise requests.ConnectionError("down")

        with patch.object(web, "http_get", boom), self.assertRaises(web.WebError):
            web.fetch("http://example.com/")

        self.assertFalse(getattr(web._pins, "map", None))

    def test_pins_belong_to_one_thread(self):
        import socket
        import threading

        seen = []

        with patch.object(web, "_real_getaddrinfo", return_value=[(socket.AF_INET, 1, 6, "", ("10.9.9.9", 80))]):
            with web._pinned("example.com", ["93.184.216.34"]):
                worker = threading.Thread(target=lambda: seen.append(socket.getaddrinfo("example.com", 80)[0][4][0]))
                worker.start()
                worker.join()

        self.assertEqual(seen, ["10.9.9.9"], "another thread's lookups must not be affected")

    def test_ipv6_addresses_and_the_family_filter(self):
        import socket

        with web._pinned("example.com", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]):
            both = socket.getaddrinfo("example.com", 443)
            self.assertEqual([f[0] for f in both], [socket.AF_INET, socket.AF_INET6])
            self.assertEqual(both[1][4], ("2606:2800:220:1:248:1893:25c8:1946", 443, 0, 0))
            only4 = socket.getaddrinfo("example.com", 443, socket.AF_INET)
            self.assertEqual([f[0] for f in only4], [socket.AF_INET])

            with self.assertRaises(socket.gaierror):
                web._pinned_getaddrinfo("example.com", 443, 99)


class FetchTests(WebTestCase):
    def test_a_redirect_to_an_internal_address_is_blocked(self):
        self.serve({"http://example.com/": page("", status=302, location="http://127.0.0.1:8080/secret")})

        with self.assertRaises(web.WebError) as caught:
            web.fetch("http://example.com/")

        self.assertIn("blocked", str(caught.exception))
        self.assertEqual(self.requested, ["http://example.com/"], "the internal address must never be requested")

    def test_relative_redirects_are_followed(self):
        self.serve({
            "http://example.com/old": page("", status=301, location="/new"),
            "http://example.com/new": page("<p>hello</p>"),
        })  # fmt: skip
        url, _, body = web.fetch("http://example.com/old")
        self.assertEqual(url, "http://example.com/new")
        self.assertIn(b"hello", body)

    def test_a_redirect_loop_stops(self):
        self.serve({"http://example.com/": page("", status=302, location="http://example.com/")})

        with self.assertRaises(web.WebError) as caught:
            web.fetch("http://example.com/")

        self.assertIn("too many redirects", str(caught.exception))

    def test_error_statuses_are_reported(self):
        self.serve({"http://example.com/": page("nope", status=404)})

        with self.assertRaises(web.WebError) as caught:
            web.fetch("http://example.com/")

        self.assertIn("404", str(caught.exception))

    def test_binary_content_is_refused(self):
        self.serve({"http://example.com/": page("%PDF", content_type="application/pdf")})

        with self.assertRaises(web.WebError) as caught:
            web.fetch("http://example.com/")

        self.assertIn("application/pdf", str(caught.exception))

    def test_huge_pages_are_cut(self):
        self.serve({"http://example.com/": FakeResponse(b"x" * (web.MAX_PAGE_BYTES + 500_000), content_type="text/plain")})
        _, _, body = web.fetch("http://example.com/")
        self.assertEqual(len(body), web.MAX_PAGE_BYTES)

    def test_network_errors_become_messages(self):
        import requests

        def boom(url):
            raise requests.ConnectionError("down")

        with patch.object(web, "http_get", boom), self.assertRaises(web.WebError) as caught:
            web.fetch("http://example.com/")

        self.assertIn("ConnectionError", str(caught.exception))

    def test_the_connection_is_closed(self):
        response = page("<p>ok</p>")
        self.serve({"http://example.com/": response})
        web.fetch("http://example.com/")
        self.assertTrue(response.closed)


ARTICLE = """<html><head><title> My  Guide </title><script>var secret = 1;</script></head>
<body><nav>Home | About | Login</nav>
<article><h1>Using gather</h1><p>{body}</p><p>The return_exceptions flag collects errors instead of raising.</p></article>
<aside>Related: ads ads ads</aside><footer>Copyright footer text</footer></body></html>""".format(body="Some intro text. " * 40)


class HtmlToTextTests(unittest.TestCase):
    def test_menus_scripts_and_footers_are_removed(self):
        title, text = web.html_to_text(ARTICLE.encode())
        self.assertEqual(title, "My Guide")
        self.assertIn("The return_exceptions flag", text)
        for junk in ["secret", "Home | About", "ads ads", "Copyright footer"]:
            self.assertNotIn(junk, text)

    def test_paragraphs_stay_on_separate_lines(self):
        _, text = web.html_to_text(b"<body><p>one</p><p>two</p><ul><li>three</li><li>four</li></ul></body>")
        self.assertEqual(text.splitlines(), ["one", "two", "three", "four"])

    def test_a_page_without_an_article_uses_the_body(self):
        _, text = web.html_to_text(b"<html><body><div>plain body text</div></body></html>")
        self.assertEqual(text, "plain body text")

    def test_garbage_does_not_crash(self):
        self.assertEqual(web.html_to_text(b""), ("", ""))
        self.assertEqual(web.html_to_text(b"   "), ("", ""))


class ReadTests(WebTestCase):
    def test_a_page_is_read_with_the_untrusted_label(self):
        self.serve({"http://example.com/g": page(ARTICLE)})
        result = web.read("http://example.com/g")
        self.assertTrue(result.startswith(web.UNTRUSTED))
        self.assertIn("Title: My Guide", result)
        self.assertIn("URL: http://example.com/g", result)

    def test_focus_returns_only_matching_passages(self):
        filler = "".join(f"<p>Filler paragraph {i} with unrelated words to pad things out here.</p>" for i in range(40))
        self.serve({"http://example.com/g": page(f"<body>{filler}<p>The return_exceptions flag matters.</p>{filler}</body>")})
        result = web.read("http://example.com/g", focus="return_exceptions", max_chars=600)
        self.assertIn("The return_exceptions flag matters.", result)
        self.assertLess(len(result), 1800)
        self.assertIn('Passages about "return_exceptions"', result)

    def test_long_pages_come_in_parts_with_a_hint(self):
        self.serve({"http://example.com/t": FakeResponse(("0123456789" * 1000).encode(), content_type="text/plain")})
        first = web.read("http://example.com/t", max_chars=1000)
        self.assertIn("Call again with start=1000", first)
        second = web.read("http://example.com/t", start=1000, max_chars=1000)
        self.assertIn("start=2000", second)
        last = web.read("http://example.com/t", start=9500, max_chars=1000)
        self.assertNotIn("Call again", last)

    def test_the_same_page_is_fetched_only_once(self):
        self.serve({"http://example.com/g": page(ARTICLE)})
        web.read("http://example.com/g")
        web.read("http://example.com/g", focus="gather")
        web.read("http://example.com/g", start=100)
        self.assertEqual(self.requested, ["http://example.com/g"])

    def test_problems_are_returned_as_error_text_not_raised(self):
        self.serve({"http://example.com/missing": page("", status=404)})
        self.assertTrue(web.read("http://example.com/missing").startswith("ERROR:"))
        self.assertTrue(web.read("http://localhost/").startswith("ERROR:"))
        self.assertTrue(web.read("not a url").startswith("ERROR:"))

    def test_a_page_with_no_text_says_so(self):
        self.serve({"http://example.com/js": page("<html><body><script>app()</script></body></html>")})
        self.assertIn("no readable text", web.read("http://example.com/js"))

    def test_an_unexpected_failure_is_reported_not_raised(self):
        with patch.object(web, "_page", side_effect=RuntimeError("bug")):
            self.assertIn("RuntimeError", web.read("http://example.com/"))

    def test_it_can_be_switched_off(self):
        with patch.dict("os.environ", {"WEB_SEARCH_ENABLED": "false"}):
            self.assertIn("turned off", web.read("http://example.com/"))

    def test_the_size_asked_for_is_limited(self):
        self.serve({"http://example.com/t": FakeResponse(b"y" * 50_000, content_type="text/plain")})
        result = web.read("http://example.com/t", max_chars=10**9)
        self.assertLess(len(result), web.MAX_PAGE_CHARS + 800)


def results(*items):
    return [{"title": t, "href": u, "body": b} for t, u, b in items]


class SearchTests(WebTestCase):
    def search(self, found, query="python asyncio", count=5):
        with patch.object(web, "_search_backend", return_value=found) as backend:
            return web.search(query, count), backend

    def test_results_are_numbered_with_title_link_and_snippet(self):
        text, _ = self.search(results(("Asyncio guide", "https://a.example/x", "How gather works."), ("Docs", "https://b.example/y", "Reference.")))
        self.assertTrue(text.startswith(web.UNTRUSTED))
        self.assertIn("1. Asyncio guide\n   https://a.example/x\n   How gather works.", text)
        self.assertIn("2. Docs", text)
        self.assertIn("read_webpage", text)

    def test_one_site_cannot_fill_the_list(self):
        found = results(*[(f"Page {i}", f"https://same.example/{i}", "x") for i in range(6)], ("Other", "https://other.example/", "y"))
        text, _ = self.search(found, count=5)
        self.assertEqual(text.count("same.example"), web.MAX_PER_DOMAIN)
        self.assertIn("other.example", text)

    def test_duplicate_links_are_dropped(self):
        text, _ = self.search(results(("A", "https://a.example/1", ""), ("A again", "https://a.example/1", "")))
        self.assertEqual(text.count("https://a.example/1"), 1)

    def test_the_number_of_results_is_respected_and_limited(self):
        found = results(*[(f"T{i}", f"https://s{i}.example/", "") for i in range(20)])
        text, backend = self.search(found, count=3)
        self.assertEqual(text.count("https://s"), 3)
        self.assertEqual(backend.call_args.args[1], 6, "asks for extra to allow for dropped duplicates")
        text, _ = self.search(found, count=999)
        self.assertEqual(text.count("https://s"), web.MAX_SEARCH_RESULTS)

    def test_long_snippets_are_cut(self):
        text, _ = self.search(results(("T", "https://a.example/", "word " * 500)))
        self.assertLess(len(text), 900)

    def test_the_same_query_is_searched_once(self):
        with patch.object(web, "_search_backend", return_value=results(("T", "https://a.example/", "s"))) as backend:
            web.search("same query")
            web.search("Same Query")
            web.search("another query")

        self.assertEqual(backend.call_count, 2)

    def test_no_results(self):
        text, _ = self.search([], query="zzzz")
        self.assertEqual(text, "No web results for: zzzz")

    def test_an_empty_query_is_an_error(self):
        self.assertTrue(web.search("   ").startswith("ERROR"))

    def test_a_failing_search_is_an_error_message_and_is_not_cached(self):
        with patch.object(web, "_search_backend", side_effect=RuntimeError("rate limited")) as backend:
            self.assertIn("web search failed", web.search("q"))
            web.search("q")

        self.assertEqual(backend.call_count, 2)

    def test_a_missing_package_is_explained(self):
        with patch.object(web, "_search_backend", side_effect=ImportError):
            self.assertIn("pip install ddgs", web.search("q"))

    def test_it_can_be_switched_off(self):
        with patch.dict("os.environ", {"WEB_SEARCH_ENABLED": "false"}):
            self.assertIn("turned off", web.search("q"))


class ToolRegistrationTests(unittest.TestCase):
    def test_the_tools_exist_and_work_through_the_registry(self):
        with patch.object(web, "_search_backend", return_value=results(("T", "https://a.example/", "s"))):
            web._searches._items.clear()
            output = tools.TOOL_BY_NAME["web_search"].invoke({"query": "registry check"})

        self.assertIn("https://a.example/", output)
        self.assertIn("read_webpage", tools.TOOL_BY_NAME)

    def test_reading_the_web_does_not_need_approval(self):
        self.assertNotIn("web_search", tools.NEEDS_APPROVAL)
        self.assertNotIn("read_webpage", tools.NEEDS_APPROVAL)


if __name__ == "__main__":
    unittest.main()
