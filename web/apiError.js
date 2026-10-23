/**
 * Unified API error type.
 * Normalized from legacy { error: "string" }, { error_msg }, or V1 { error: { code, message, details } }.
 */
export class ApiError extends Error {
    constructor({ code = 'unknown', message = 'Unknown error', status = 0, details = null, requestId = null } = {}) {
        super(message);
        this.name = 'ApiError';
        this.code = code;
        this.message = message;
        this.status = status;
        this.details = details;
        this.requestId = requestId;
    }

    toString() {
        return `ApiError(${this.code}): ${this.message}`;
    }
}
