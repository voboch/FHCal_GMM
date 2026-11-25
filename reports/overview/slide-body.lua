-- Group native Markdown content between the fixed heading and the slide's sources.
local function role(block, name)
  return block.t == "Div" and block.classes:includes(name)
end

function Pandoc(document)
  if not FORMAT:match("revealjs") then
    return document
  end

  local output = pandoc.List()
  local body = nil
  local footnotes = nil
  local notes = pandoc.List()

  local function finish_slide()
    if body == nil then
      return
    end
    output:insert(pandoc.Div(body, pandoc.Attr("", {"slide-body"})))
    if footnotes ~= nil then
      output:insert(footnotes)
    end
    output:extend(notes)
    body, footnotes, notes = nil, nil, pandoc.List()
  end

  for _, block in ipairs(document.blocks) do
    if block.t == "Header" and block.level <= 2 then
      finish_slide()
      output:insert(block)
      if block.level == 2 and not block.classes:includes("deck-cover") then
        body = pandoc.List()
      end
    elseif body ~= nil then
      if role(block, "notes") then
        notes:insert(block)
      elseif role(block, "slide-footnotes") then
        assert(footnotes == nil, "Use one slide-footnotes block per slide")
        footnotes = block
      else
        body:insert(block)
      end
    else
      output:insert(block)
    end
  end

  finish_slide()
  document.blocks = output
  return document
end
