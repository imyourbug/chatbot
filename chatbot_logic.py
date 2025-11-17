from enum import Enum
from typing import Dict, List, Literal, Optional
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

class CustomerRequirement(str, Enum):
    CONFIRM_PRODUCT = "confirm_product"
    REQUEST_PRODUCT_INFO = "request_product_info"
    PROVIDE_SIZE = "provide_size"
    REQUEST_POLICY = "request_policy"
    REQUEST_PRICE = "request_price"
    PROVIDE_LOCATION = "provide_location"
    PROVIDE_CONTACT = "provide_contact"
    CONFIRM_ORDER = "confirm_order"
    ASK_PRODUCT_LINE = "ask_product_line"
    CONFIRM_PRODUCT_LINE = "confirm_product_line"
    OTHER = "other"


class ChatbotAction(str, Enum):
    SEND_PRODUCT_IMAGE = "send_product_image"
    SEND_PRODUCT_INFO = "send_product_info"
    ASK_FOR_SIZE = "ask_for_size"
    SEND_POLICY = "send_policy"
    SEND_PRICE_WITH_GIFT = "send_price_with_gift"
    ASK_FOR_LOCATION = "ask_for_location"
    ASK_FOR_CONTACT = "ask_for_contact"
    CONFIRM_ORDER = "confirm_order"
    ASK_PRODUCT_LINE = "ask_product_line"
    SEND_BEST_SELLER = "send_best_seller"
    FALLBACK = "fallback"


INTENT_TO_ACTION: Dict[CustomerRequirement, ChatbotAction] = {
    CustomerRequirement.CONFIRM_PRODUCT: ChatbotAction.SEND_PRODUCT_IMAGE,
    CustomerRequirement.REQUEST_PRODUCT_INFO: ChatbotAction.SEND_PRODUCT_INFO,
    CustomerRequirement.PROVIDE_SIZE: ChatbotAction.ASK_FOR_SIZE,
    CustomerRequirement.REQUEST_POLICY: ChatbotAction.SEND_POLICY,
    CustomerRequirement.REQUEST_PRICE: ChatbotAction.SEND_PRICE_WITH_GIFT,
    CustomerRequirement.PROVIDE_LOCATION: ChatbotAction.ASK_FOR_LOCATION,
    CustomerRequirement.PROVIDE_CONTACT: ChatbotAction.ASK_FOR_CONTACT,
    CustomerRequirement.CONFIRM_ORDER: ChatbotAction.CONFIRM_ORDER,
    CustomerRequirement.ASK_PRODUCT_LINE: ChatbotAction.ASK_PRODUCT_LINE,
    CustomerRequirement.CONFIRM_PRODUCT_LINE: ChatbotAction.SEND_BEST_SELLER,
    CustomerRequirement.OTHER: ChatbotAction.FALLBACK,
}


def build_intent_prompt(conversation: List[Dict[Literal["role", "content"], str]]) -> str:
    instructions = """
Bạn là trợ lý phân loại yêu cầu khách hàng cho shop thời trang.
Nhiệm vụ: đọc đoạn hội thoại (theo thời gian) và xác định ý định hiện tại của khách.
Chỉ trả về đúng 1 intent trong danh sách sau:
confirm_product, request_product_info, provide_size, request_policy, request_price,
provide_location, provide_contact, confirm_order, ask_product_line,
confirm_product_line, other.
Nếu không chắc chắn hãy trả về other.
Hội thoại:
"""
    serialized_history = "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)
    return f"{instructions.strip()}\n\n{serialized_history.strip()}\n\nintent:"


class ChatbotIntentResolver:
    def __init__(
        self,
        client: Optional["OpenAI"] = None,
        intent_model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client or OpenAI()
        self._intent_model = intent_model

    @property
    def client(self) -> "OpenAI":
        if not self._client:
            raise RuntimeError("OpenAI client is not available.")
        return self._client

    def classify_intent(
        self,
        conversation: List[Dict[Literal["role", "content"], str]],
        intent_model: Optional[str] = None,
    ) -> CustomerRequirement:
        model = intent_model or self._intent_model
        prompt = build_intent_prompt(conversation)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "Phân loại intent"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
        )
        intent = response.choices[0].message.content  # type: ignore[index]
        return (
            CustomerRequirement(intent)
            if intent in CustomerRequirement._value2member_map_
            else CustomerRequirement.OTHER
        )

    def determine_action(
        self,
        conversation: List[Dict[Literal["role", "content"], str]],
        intent_model: Optional[str] = None,
    ) -> ChatbotAction:
        intent = self.classify_intent(conversation, intent_model=intent_model)
        return INTENT_TO_ACTION.get(intent, ChatbotAction.FALLBACK)


__all__ = [
    "CustomerRequirement",
    "ChatbotAction",
    "INTENT_TO_ACTION",
    "build_intent_prompt",
    "ChatbotIntentResolver",
]

