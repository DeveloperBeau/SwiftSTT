import Foundation
import Synchronization

/// Answers every request in a test session without touching the network.
///
/// It claims every request, so nothing a stubbed session sends leaves the machine.
/// `ggml-tiny.bin` gets ``modelBytes`` with status 200 and every other path gets 404.
/// Each answer is held back for a short delay, so a caller that stops waiting early
/// looks for the model before it has landed.
final class StubURLProtocol: URLProtocol, @unchecked Sendable {

    /// The bytes served for `ggml-tiny.bin`.
    static let modelBytes = Data("stub tiny model bytes".utf8)

    /// The delay before each answer.
    static let answerDelay = DispatchTimeInterval.milliseconds(200)

    private static let requestCounts = Mutex<[String: Int]>([:])

    /// Forgets every request counted so far.
    static func reset() {
        requestCounts.withLock { $0 = [:] }
    }

    /// The number of requests whose URL ends in `fileName`.
    static func requestCount(for fileName: String) -> Int {
        requestCounts.withLock { $0[fileName, default: 0] }
    }

    /// Every file name requested so far, with its count.
    static var allRequestCounts: [String: Int] {
        requestCounts.withLock { $0 }
    }

    /// An ephemeral session whose requests are all answered by this stub.
    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let fileName = url.lastPathComponent
        Self.requestCounts.withLock { $0[fileName, default: 0] += 1 }
        let isModel = fileName == "ggml-tiny.bin"
        let body = isModel ? Self.modelBytes : Data()
        let response = HTTPURLResponse(
            url: url, statusCode: isModel ? 200 : 404, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": "\(body.count)"])
        DispatchQueue.global().asyncAfter(deadline: .now() + Self.answerDelay) {
            [self] in
            if let response {
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            }
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}
