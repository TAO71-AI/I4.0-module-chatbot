# Import libraries
import logging
from typing import Any
from collections.abc import Generator
import re
import base64
import json
import copy
import EnabledModules.chatbot.llama_utils as utils_llama

SERVER_VERSION_MIN: int = 170000
SERVER_VERSION_MAX: int = 9999999

__models__: dict[str, dict[str, Any]] = {}
ServiceConfiguration: dict[str, Any] | None = None
ServerConfiguration: dict[str, Any] | None = None

def __check_service_configuration__() -> None:
    if (ServiceConfiguration is None):
        raise ValueError("Service configuration is not defined.")
    
    if (ServerConfiguration is None):
        raise ValueError("Server configuration is not defined.")

def SERVICE_LOAD_MODELS(Models: dict[str, dict[str, Any]]) -> None:
    """
    Load all the chatbot models.

    Args:
        Models (dict[str, dict[str, Any]]): All the models to load.
    
    Returns:
        None
    """
    for name, configuration in Models.items():
        LoadModel(name, configuration)

def SERVICE_OFFLOAD_MODELS(Names: list[str]) -> None:
    """
    Offload all the defined chatbot models.

    Args:
        Names (list[str]): Names of the models to offload.
    
    Returns:
        None
    """
    # Define globals
    global __models__

    # Check configuration
    __check_service_configuration__()
    
    for name in Names:
        # Make sure the model is loaded
        if (__models__[name]["_private_model"] is None):
            continue
        
        logging.info("[service_chatbot] Offloading model.")

        # Offload the model
        if (__models__[name]["_private_type"] == "lcpp"):
            if (__models__[name]["_private_model"].chat_handler is not None):
                __models__[name]["_private_model"].chat_handler.close()
                __models__[name]["_private_model"].chat_handler = None
            
            __models__[name]["_private_model"].close()
        
        __models__[name]["_private_model"] = None

def SERVICE_INFERENCE(Name: str, UserConfig: dict[str, Any], UserParameters: dict[str, Any]) -> Generator[dict[str, Any]]:
    """
    Inference the chatbot model.

    Args:
        Name (str): Name of the model.
        UserConfig (dict[str, Any]): Configuration of the user.
        UserParameters (dict[str, Any]): Parameters of the user ("key_info", "conversation").
    """
    __check_service_configuration__()
    conversation = UserParameters["conversation"]

    temperature = UserConfig["temperature"] if ("temperature" in UserConfig and ServiceConfiguration["temperature"]["modified_by_user"]) else __models__[Name]["temperature"] if ("temperature" in __models__[Name]) else ServiceConfiguration["temperature"]["default"]
    topP = UserConfig["top_p"] if ("top_p" in UserConfig and ServiceConfiguration["top_p"]["modified_by_user"]) else __models__[Name]["top_p"] if ("top_p" in __models__[Name]) else ServiceConfiguration["top_p"]["default"]
    topK = UserConfig["top_k"] if ("top_k" in UserConfig and ServiceConfiguration["top_k"]["modified_by_user"]) else __models__[Name]["top_k"] if ("top_k" in __models__[Name]) else ServiceConfiguration["top_k"]["default"]
    topNSigma = UserConfig["top_n_sigma"] if ("top_n_sigma" in UserConfig and ServiceConfiguration["top_n_sigma"]["modified_by_user"]) else __models__[Name]["top_n_sigma"] if ("top_n_sigma" in __models__[Name]) else ServiceConfiguration["top_n_sigma"]["default"]
    minP = UserConfig["min_p"] if ("min_p" in UserConfig and ServiceConfiguration["min_p"]["modified_by_user"]) else __models__[Name]["min_p"] if ("min_p" in __models__[Name]) else ServiceConfiguration["min_p"]["default"]
    typicalP = UserConfig["typical_p"] if ("typical_p" in UserConfig and ServiceConfiguration["typical_p"]["modified_by_user"]) else __models__[Name]["typical_p"] if ("typical_p" in __models__[Name]) else ServiceConfiguration["typical_p"]["default"]
    seed = UserConfig["seed"] if ("seed" in UserConfig and ServiceConfiguration["seed"]["modified_by_user"]) else __models__[Name]["seed"] if ("seed" in __models__[Name]) else ServiceConfiguration["seed"]["default"]
    presencePenalty = UserConfig["presence_penalty"] if ("presence_penalty" in UserConfig and ServiceConfiguration["presence_penalty"]["modified_by_user"]) else __models__[Name]["presence_penalty"] if ("presence_penalty" in __models__[Name]) else ServiceConfiguration["presence_penalty"]["default"]
    frequencyPenalty = UserConfig["frequency_penalty"] if ("frequency_penalty" in UserConfig and ServiceConfiguration["frequency_penalty"]["modified_by_user"]) else __models__[Name]["frequency_penalty"] if ("frequency_penalty" in __models__[Name]) else ServiceConfiguration["frequency_penalty"]["default"]
    repeatPenalty = UserConfig["repeat_penalty"] if ("repeat_penalty" in UserConfig and ServiceConfiguration["repeat_penalty"]["modified_by_user"]) else __models__[Name]["repeat_penalty"] if ("repeat_penalty" in __models__[Name]) else ServiceConfiguration["repeat_penalty"]["default"]
    penaltyLastN = UserConfig["penalty_last_n"] if ("penalty_last_n" in UserConfig and ServiceConfiguration["penalty_last_n"]["modified_by_user"]) else __models__[Name]["penalty_last_n"] if ("penalty_last_n" in __models__[Name]) else ServiceConfiguration["penalty_last_n"]["default"]
    tools = UserConfig["tools"] if ("tools" in UserConfig and ServiceConfiguration["tools"]["modified_by_user"]) else []
    toolChoice = UserConfig["tool_choice"] if ("tool_choice" in UserConfig and ServiceConfiguration["tool_choice"]["modified_by_user"]) else "auto" if (len(tools) > 0) else "none"
    maxLength = UserConfig["max_length"] if ("max_length" in UserConfig and ServiceConfiguration["max_length"]["modified_by_user"]) else __models__[Name]["max_length"] if ("max_length" in __models__[Name]) else ServiceConfiguration["max_length"]["default"]
    stopTokens = UserConfig.get("stop_tokens", UserConfig.get("stop", []))
    stopTokens = [str(s) for s in stopTokens] if (isinstance(stopTokens, list)) else [str(stopTokens)]
    extraParameters = __models__[Name].get("_private_extra_parameters", {})
    
    if (maxLength > ServiceConfiguration["max_length"]["default"] and not ServiceConfiguration["max_length"]["allow_greater_than_default"]):
        maxLength = ServiceConfiguration["max_length"]["default"]
    
    extraChatTemplateParams = __models__[Name].get("_private_chat_template_params", {})
    startToken = None
    endToken = None
    modes = __models__[Name].get("modes", {})
    userModes = UserConfig.get("modes", None)
    defaultChannel = None

    if (isinstance(userModes, str)):
        userModes = [userModes]
    elif (not isinstance(userModes, list)):
        if (userModes is not None):
            yield {"warnings": ["Specified mode type is not valid. Setting to default."]}
        
        userModes = __models__[Name].get("default_modes", [])

    for modeName, modeParams in modes.items():
        if (modeName not in userModes):
            continue

        for paramName, paramValue in modeParams.items():
            if (paramName == "conflicts"):
                if (isinstance(paramValue, str)):
                    paramValue = [paramValue]
                elif (not isinstance(paramValue, list)):
                    logging.warning("[service_chatbot] Invalid mode param value. Ignoring.")
                    continue
                
                for conflictsModes in paramValue:
                    if (conflictsModes in userModes):
                        raise ValueError("Conflict in modes.")
            elif (paramName in ["chat_template_params", "_private_chat_template_params"]):
                if (not isinstance(paramValue, dict)):
                    logging.warning("[service_chatbot] Invalid mode param value. Ignoring.")
                    continue

                for ctParamName, ctParamValue in paramValue.items():
                    extraChatTemplateParams[ctParamName] = ctParamValue
            elif (paramName in ["start_token", "_private_start_token"]):
                startToken = paramValue
            elif (paramName in ["end_token", "_private_end_token"]):
                endToken = paramValue
            elif (paramName == "start_channel"):
                defaultChannel = paramValue
            else:
                logging.warning("[service_chatbot] Unknown mode param. Ignoring.")
                continue

    continueGeneration = conversation[-1]["role"] == "assistant"
    
    generator = InferenceModel(
        Name,
        conversation,
        {
            "temperature": temperature,
            "top_p": topP,
            "top_k": topK,
            "top_n_sigma": topNSigma,
            "min_p": minP,
            "typical_p": typicalP,
            "seed": seed,
            "presence_penalty": presencePenalty,
            "frequency_penalty": frequencyPenalty,
            "repeat_penalty": repeatPenalty,
            "penalty_last_n": penaltyLastN,
            "tools": tools,
            "tool_choice": toolChoice,
            "max_length": maxLength,
            "stop": stopTokens,
            "start_token": startToken,
            "end_token": endToken,
            "chat_template_params": extraChatTemplateParams,
            "extra_parameters": extraParameters,
            "continue_generation": continueGeneration,
            "default_channel": defaultChannel
        }
    )

    for token in generator:
        yield token

def InferenceModel(Name: str, Conversation: list[dict[str, str | list[dict[str, str]]]], Configuration: dict[str, Any]) -> Generator[dict[str, Any]]:
    """
    Inference the model.

    Args:
        Name (str): Name of the model.
        Conversation (list[dict[str, str | list[dict[str, str]]]]): Conversation of the model.
        Configuration (dict[str, Any]): Configuration of the model.
    """
    LoadModel(Name, __models__[Name])

    conversation = copy.deepcopy(Conversation)
    modelConversation = []
    replaceRoles = __models__[Name].get("_private_replace_roles", ServiceConfiguration["replace_roles"])
    
    for message in conversation:
        if (message["role"] == "system"):
            content = ""

            for cont in message["content"]:
                if (cont["type"] != "text"):
                    raise TypeError("Invalid content type for system prompt. Only text allowed.")
                    
                content += cont["text"]

            message["content"] = content

        if (isinstance(message["content"], list)):
            txt = None

            for content in message["content"]:
                if (content["type"] not in __models__[Name]["multimodal"]):
                    yield {"warnings": [f"Content type '{content['type']}' not supported by this model, this will be ignored."]}
                    continue
                elif (len(__models__[Name]["multimodal"]) == 1 and __models__[Name]["multimodal"][0] == "text"):
                    if (txt is None):
                        txt = content["text"]
                    else:
                        txt += content["text"]

                if (__models__[Name]["_private_type"] == "lcpp"):
                    if (content["type"] == "image"):
                        content["image"] = f"data:image;base64,{content['image']}"
                    elif (content["type"] == "audio"):
                        content["audio"] = f"data:audio;base64,{content['audio']}"
                    elif (content["type"] == "video"):
                        content["video"] = f"data:video;base64,{content['video']}"
                    elif (content["type"] != "text"):
                        yield {"warnings": ["Unsupported media type, will be ignored."]}
                        continue
            
            if (txt is not None):
                message["content"] = txt
        
        message["role"] = replaceRoles.get(message["role"], message["role"])
        modelConversation.append(message)
    
    tools = []
    currentToolIdx = None
    toolStartToken = __models__[Name].get("tool_start_token", ServiceConfiguration["tool_start_token"])
    toolEndToken = __models__[Name].get("tool_end_token", ServiceConfiguration["tool_end_token"])

    defaultChannel = Configuration.get("default_channel", None)
    channelStartToken = __models__[Name].get("channel_start_token", ServiceConfiguration["channel_start_token"])
    channelEndToken = __models__[Name].get("channel_end_token", ServiceConfiguration["channel_end_token"])
    channelNameEndToken = __models__[Name].get("channel_name_end_token", ServiceConfiguration["channel_name_end_token"])
    channelName = defaultChannel
    prevChannelName = channelName
    settingChannelName = False

    if (defaultChannel is None):
        defaultChannel = __models__[Name].get("channel_default", ServiceConfiguration["channel_default"])

    fullAssistantText = ""
    firstToken = True
    startToken = Configuration["start_token"]
    endToken = Configuration["end_token"]

    if (__models__[Name]["_private_type"] == "lcpp"):
        model: utils_llama.Llama = __models__[Name]["_private_model"]
        prevChatHandlerTemplateArgs = None

        if (model.chat_handler is None):
            pass  # TODO
        else:
            prevChatHandlerTemplateArgs = copy.deepcopy(model.chat_handler.extra_template_arguments)
            model.chat_handler.extra_template_arguments |= Configuration["chat_template_params"]

        response = model.create_chat_completion(
            messages = modelConversation,
            tools = Configuration["tools"],
            tool_choice = Configuration["tool_choice"],
            temperature = Configuration["temperature"],
            top_p = Configuration["top_p"],
            top_k = Configuration["top_k"],
            top_n_sigma = Configuration["top_n_sigma"],
            min_p = Configuration["min_p"],
            typical_p = Configuration["typical_p"],
            stream = True,
            seed = Configuration["seed"],
            max_tokens = Configuration["max_length"],
            present_penalty = Configuration["presence_penalty"],
            frequency_penalty = Configuration["frequency_penalty"],
            repeat_penalty = Configuration["repeat_penalty"],
            penalty_last_n = Configuration["penalty_last_n"],
            stop = Configuration["stop"],
            assistant_prefill = Configuration["continue_generation"] and ServiceConfiguration["allow_assistant_prefill"],
            **Configuration["extra_parameters"]
        )
    
    try:
        if (startToken is not None):
            yield {"text": startToken}

        for token in response:
            tokenText = ""

            if (__models__[Name]["_private_type"] == "lcpp"):
                if ("choices" not in token or len(token["choices"]) == 0 or "delta" not in token["choices"][0] or "content" not in token["choices"][0]["delta"]):
                    continue

            tokenText: str = token["choices"][0]["delta"]["content"]
            fullAssistantText += tokenText

            if (settingChannelName):
                if (channelNameEndToken is None or channelNameEndToken in tokenText):
                    channelName += tokenText[:tokenText.index(channelNameEndToken)] if (channelNameEndToken is not None) else __models__[Name].get("channel_null_name", ServiceConfiguration["channel_null_name"])
                    settingChannelName = False

                    if (channelName != prevChannelName):
                        yield {"text": __handle_channel_change__(Name, prevChannelName, channelName)}
                    
                    prevChannelName = channelName
                else:
                    channelName += tokenText
                
                continue

            if (channelStartToken in tokenText):
                textBeforeToken = tokenText[:tokenText.index(channelStartToken)]

                if (len(textBeforeToken) > 0):
                    yield {"text": textBeforeToken, "extra": {"channel": prevChannelName}}
                
                prevChannelName = channelName
                channelName = tokenText[tokenText.index(channelStartToken) + len(channelStartToken):]
                settingChannelName = True

                continue
            
            if (channelEndToken in tokenText):
                if (channelName != defaultChannel):
                    yield {"text": __handle_channel_change__(Name, channelName, prevChannelName)}
                
                channelName = defaultChannel
                prevChannelName = channelName

                continue

            if (toolEndToken in tokenText and currentToolIdx is not None):
                tools[currentToolIdx] += tokenText[:tokenText.index(toolEndToken)]
                currentToolIdx = None

            if (toolStartToken in tokenText and currentToolIdx is None):
                currentToolIdx = len(tools)
                tools.append(tokenText[tokenText.index(toolStartToken) + len(toolStartToken):])
            elif (currentToolIdx is not None):
                tools[currentToolIdx] += tokenText

            firstToken = False
            yield {"text": tokenText, "extra": {"channel": channelName}}
        
        if (endToken is not None):
            yield {"text": endToken}
    finally:
        if (__models__[Name]["_private_type"] == "lcpp"):
            if (prevChatHandlerTemplateArgs is not None):
                if (model.chat_handler is None):
                    pass  # TODO
                else:
                    model.chat_handler.extra_template_arguments = prevChatHandlerTemplateArgs
                    prevChatHandlerTemplateArgs = None
            
            if ("_private_delete_kv_cache" in __models__[Name] and __models__[Name]["_private_delete_kv_cache"]):
                utils_llama.ClearLlamaCache(model)

    parsedTools = []
    toolsType = __models__[Name].get("tool_parse_type", None)

    if (toolsType is None):
        if (model.metadata["general.architecture"] in ["qwen35moe", "qwen35"]):
            toolsType = "qwen3.5"
        elif (model.metadata["general.architecture"] in ["diffusion-gemma", "gemma4"]):
            toolsType = "gemma4"
        else:
            toolsType = "json-1"
    
    if (toolsType in ["json", "json-1"]):
        parsedTools = [json.loads(tool) for tool in tools]
    elif (toolsType in ["xml", "xml-1", "qwen3.5"]):
        for tool in tools:
            toolName = tool.strip()
            toolName = toolName[toolName.index("<function"):]
            toolName = toolName[toolName.index("=") + 1:].strip()

            if (toolName.startswith("\"")):
                toolName = toolName[1:toolName.index("\"")]
            elif (toolName.startswith("\'")):
                toolName = toolName[1:toolName.index("\'")]
            else: 
                toolName = toolName[:toolName.index(">")]

            toolName = toolName.strip()

            toolParams = tool[tool.index(toolName) + len(toolName):].strip()
            toolParams = toolParams[toolParams.index(">") + 1:toolParams.index("</function>")].strip()
            toolParams = toolParams.split("<parameter")[1:] if ("<parameter" in toolParams) else []

            paramsParsed = {}

            for param in toolParams:
                paramName = param.strip()
                paramName = paramName[paramName.index("=") + 1:].strip()

                if (paramName.startswith("\"")):
                    paramName = paramName[1:paramName.index("\"")]
                elif (paramName.startswith("\'")):
                    paramName = paramName[1:paramName.index("\'")]
                else:
                    paramName = paramName[:paramName.index(">")]

                paramName = paramName.strip()

                paramValue = param[param.index(paramName) + len(paramName):].strip()
                paramValue = paramValue[paramValue.index(">") + 1:paramValue.index("</parameter>")].strip()

                try:
                    paramValue = json.loads(paramValue)
                except json.JSONDecodeError:
                    paramValue = str(paramValue)

                paramsParsed[paramName] = paramValue
            
            parsedTools.append({"name": toolName, "arguments": paramsParsed})
    elif (toolsType in ["gemma4", "gemma-4", "diffusion-gemma", "diffusion-gemma-1"]):
        for tool in tools:
            globalPattern = r"^call:([a-zA-Z0-9_-]+)\s*\{(.+)\}$"
            coincide = re.match(globalPattern, tool.strip(), re.IGNORECASE | re.DOTALL)

            toolName = coincide.group(1)
            toolContent = coincide.group(2)

            argsPattern = r"([a-zA-Z0-9_-]+)\s*:\s*<\|\"\|>(.*?)<\|\"\|>"
            argsFound = re.findall(argsPattern, toolContent)
            paramsParsed = {}

            for paramName, paramStr in argsFound:
                paramStr = paramStr.strip()

                try:
                    paramsParsed[paramName] = json.loads(paramStr)
                except json.JSONDecodeError:
                    paramsParsed[paramName] = str(paramStr)

            parsedTools.append({"name": toolName, "arguments": paramsParsed})
    else:
        raise ValueError("Invalid tools parser.")

    yield {"extra": {"tools": parsedTools}}

def LoadModel(Name: str, Configuration: dict[str, Any]) -> None:
    """
    Load a chatbot model.

    Args:
        Name (str): Name of the model.
        Configuration (dict[str, Any]): Configuration of the model.
    
    Returns:
        None
    """
    # Define globals
    global __models__

    # Make sure the model is not loaded
    if (Name in __models__ and __models__[Name]["_private_model"] is not None):
        return
    
    # Check configuration
    __check_service_configuration__()
    
    logging.info("[service_chatbot] Loading model.")

    # Get the model type
    if ("_private_type" in Configuration):
        modelType = Configuration["_private_type"]
    else:
        modelType = None
    
    if (not isinstance(modelType, str) or (modelType != "hf" and modelType != "lcpp")):
        modelType = None
    
    if (modelType is None):
        raise AttributeError("[service_chatbot] Model type is not valid or not defined.")
    
    # Load the model
    if (modelType == "lcpp"):
        model = utils_llama.LoadLlamaModel(Configuration)

    __models__[Name] = Configuration | model

    # Test the inference
    if ("_private_test_inference" in Configuration and Configuration["_private_test_inference"]):
        logging.info("[service_chatbot] Testing inference of the model.")
        testInferenceConversation = copy.deepcopy(ServiceConfiguration["test_inference_conversation"])

        for message in testInferenceConversation:
            if (isinstance(message["content"], str)):
                message["content"] = [{"type": "text", "text": message["content"]}]
            
            for content in message["content"]:
                if (content["type"] == "text"):
                    continue

                with open(content[content["type"]], "rb") as f:  # Content type is not `text`, so it MUST be a file path
                    content[content["type"]] = base64.b64encode(f.read()).decode("utf-8")

        response = SERVICE_INFERENCE(
            Name,
            ServiceConfiguration["test_inference_configuration"],
            {"conversation": testInferenceConversation}
        )
        testInferenceResponse = ""

        for token in response:
            if ("text" in token):
                testInferenceResponse += token["text"]
        
        logging.info(f"[service_chatbot] Test inference response for model `{Name}`:\n```markdown\n{testInferenceResponse}\n```")

def __handle_channel_change__(ModelName: str, FromChannel: str, ToChannel: str) -> str:
    token = __models__[ModelName].get("channel_change_token", ServiceConfiguration["channel_change_token"])

    if (token is None):
        return
    
    if (isinstance(token, str)):
        return token
    elif (isinstance(token, dict) and FromChannel in token):
        if (isinstance(token[FromChannel], str)):
            return token[FromChannel]
        elif (isinstance(token[FromChannel], dict) and "to" in token[FromChannel]):
            return token[FromChannel].get(token[FromChannel]["to"], "\n")
    
    return ""