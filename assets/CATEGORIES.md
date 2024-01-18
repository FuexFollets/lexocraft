# Categories of Tokens and Symbols

- Component symbols are any of the symbols that are contained in this string: "~_/-'." (excluding the double quotes)

- Alphanumeric Tokens: Tokens that contain only alphanumeric characters. Can be uppercase or lowercase.

- Acronym Tokens: Tokens that only contain uppercase letters or letters that are separated by periods.

- Digit Tokens: All digits are considered digit tokens. There are no number tokens, only digit tokens. Numbers are sequences of digits that do not have spaces after them. These are not considered separate tokens if they are surrounded by letters

- Homogenous Tokens: Any sequence of characters that are not spaces where all characters are either [a-zA-Z], [0-9], or are component symbols

- Symbol Token: Any singular symbol that is not part of one of the previous tokens.

- If there is a space between any two characters, they are not part of the same token. Here are examples of tokens, splits between tokens, and examples of tokenizing strings.

- Spaces are never tokens or part of tokens

- Digits and symbols are separated

- Any token that can be found in the database is considered its own token, even if it has symbols that violate the previous rules

## Token Fields:

The following class is the data that is stored in a token. 
```cpp
class Token {
    public:

    enum class Type {
        Alphanumeric,
        Acronym,
        Digit,
        Homogeneous,
        Symbol,
    };

    std::string value;
    Type type;
    bool next_is_space;
};
```

The following c++ code can check if a token is in the database

```cpp
const bool is_in_database = database.search_from_map(token_string).has_value();
```

## Token examples:

This is the layout for each token: (value, type, next_is_space)

- "apple banana" => ("apple": Alphanumeric, true), ("banana": Alphanumeric, false)

- "123.32" => ("1": Digit, false), ("2": Digit, false), ("3": Digit, false), (".": Symbol, false), ("3": Digit, false), ("2": Digit, false)

- "apple .banana" => ("apple": Alphanumeric, false), (" ": Symbol, true), ("banana": Alphanumeric, false)
- "Can O' beans" => ("Can": Alphanumeric, true), ("O'": Alphanumeric, true), ("beans": Alphanumeric, false)
- "A.BC" => ("A": Alphanumeric, false), (".": Symbol, false), ("B": Alphanumeric, false), ("C": Alphanumeric, false)
- "A.B.C" => ("A.B.C": Acronym, false)
- "A.B.C." => ("A.B.C": Acronym, false), (".": Acronym, false)
- "9-10" => ("9": Digit, false), ("-": Symbol, false), ("1": Digit, false), ("0": Digit, false)
- "so-so" => ("so-so": Homogenous, false) // "so-so" Exists in database
- "some-of-time-time" => ("some": Alphanumeric, false), ("-": Symbol, false), ("of": Alphanumeric, false), ("-": Symbol, false), ("the": Alphanumeric, false), ("-": Symbol, false), ("time": Alphanumeric, false), ("-": Symbol, false) // "some-of-the-time" does not exist in the database
- "A'B.C-." => ("A'B": Homogenous, false), (".": Symbol, false), ("C-": Symbol: false) // Even if "A'B" and "B.C" exist in the database, "A'B" is recognized as a token since it appears first.
- "1" => ("1": Digit, false)
- "12!-4. 3 times" => ("1": Digit, false), ("2": Digit, false), ("!": Symbol, false), ("-": Symbol, false), ("4": Digit, false), (".": Symbol, true), ("3": Digit, true), ("times": Alphanumeric, false)

# Prompt:

There is a database full of recognized tokens. Based on the cases, the existence of a token existing in the token database has an effect on the tokenization. Right now, the database is simply a list of tokens WITHOUT their category. First, I need to catagorize the database. Each element in the database is NOT multiple tokens. By categorizing the database, I mean determining a category for each element in the database, which is simply a collection of short strings, words, symbols, homogenous tokens, and digits all without order. When categorizing a individual token in the database, if its category is dependent on it being in the database, always assume that it is in the database since each individual element of the database are always one token, no matter what.

Secondly, I need to transform text into an array of tokens. Given the database, it can be checked if a token is in that database from the code example above.

### Methods to Implement:

```cpp
Token::Type token_type(const std::string& value);

std::vector<Token> tokenize(const std::string& text, const VectorDatabase& vector_database);
```

The first method `token_type` will determine the `Token::Type` that a value in the database should be. `tokenize` will take a string of text and will transform it into a `std::vector` of tokens given the text and vector_database.

What should be the implementation of these two functions? Additionally, if it makes it easier, what additional functions or classes should be defined in order to make the code better?
